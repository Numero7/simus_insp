import pandas as pd
import numpy as np
import os


"""
load a realistic popularity profile from 2022 data
"""
def compute_model_from_historique(filename):
  df = pd.read_csv(filename, sep = ';')
  df["name"] = df["Ministère"] + df["Poste"]

  #create lists from panda format
  matching = df["Ministère"].tolist()

  schools = list(set(matching))
  capacites = list(map(lambda school: matching.count(school), schools))

  nb_schools = len(schools)
  affectes = [0] * nb_schools
  nb_candidats = sum( map(lambda r : str(r).isnumeric(), df["Rang"].tolist()) )
  for rang in range(nb_candidats):
    match = matching[rang]
    for school_id in range(nb_schools):
      school = schools[school_id]
      if match == school:
        affectes[school_id] = affectes[school_id] + 1
        break
  
  proposals = list(map(lambda r : set(matching[r:]) , range(nb_candidats)))
  nb_oui = list(map(lambda school : matching.count(school) , schools))

  #loop for updating popularities  
  popularities = [1] * len(schools)
  while 1:
    popularities, delta = update_popularities(popularities, proposals, nb_oui, schools)
    if delta < 0.01:
      break
  
  return schools,popularities,capacites
  
  
"""one loop of the popularity update"""
def update_popularities(pops, proposals, nb_oui, schools):
  nb_schools = len(schools)
  school_ids = range(nb_schools)
  
  #precompute inv of sum of popularities recevied by each student
  candidats = range(len(proposals))
  sum_pops = [0] * len(candidats)
  for school_id in school_ids:
    school = schools[school_id]
    pop = pops[school_id]
    for c in candidats:
      if school in proposals[c]:
         sum_pops[c] += pop
  sum_inv_pops = list(map(lambda pop : 1.0 / pop, sum_pops))

  new_pops = [0] * len(pops)
  for school_id in school_ids:
    school = schools[school_id]
    sum_inv = 0.0
    for c in candidats:
      if school in proposals[c]:
        sum_inv += sum_inv_pops[c]
    if sum_inv == 0:
      None
    new_pops[school_id] = nb_oui[school_id] / sum_inv

  #normalize  
  total_pops = sum(new_pops)
  new_pops = list(map(lambda x : nb_schools * x /total_pops, new_pops))

  delta = map(lambda rank: abs(pops[rank] - new_pops[rank]) / abs(pops[rank] + new_pops[rank]), range(0,len(pops)))

  return new_pops, max(delta)

"""convert a tsirng rank to a numeric rank"""
def to_rank(rang_str):
  if str(rang_str).isnumeric():
    return int(str(rang_str))
  else:
    return -1

if __name__ == "__main__":
  schools, popularities, capacities = compute_model_from_historique("data/2022.csv")
  
  with open("res.csv", "w") as f:
    f.write("nom;capacite;popularite\n")
    min_pop = min(popularities)
    for school_id in range(len(schools)):
      school = schools[school_id]
      pop = popularities[school_id]
      capa = capacities[school_id]
      f.write('\"' + school[:60] + '\";' + str(capa) + ";" + str(int(pop / min_pop)) + "\n")
