import pandas as pd
import numpy as np
import os
import run as run
import random as rnd
import json as json
import sys as sys

"""
load a realistic popularity profile from 2022 data
"""
def compute_model_from_historique(filename):
  df = pd.read_csv(filename, sep = ';')
  df["name"] = df["Ministère"] + df["Poste"]

  #create lists from panda format
  matching = df["Ministère"].tolist()

  employers = list(set(matching))
  capacites = list(map(lambda school: matching.count(school), employers))

  nb_employers = len(employers)
  affectes = [0] * nb_employers
  nb_candidats = sum( map(lambda r : str(r).isnumeric(), df["Rang"].tolist()) )
  for rang in range(nb_candidats):
    match = matching[rang]
    for job_id in range(nb_employers):
      school = employers[job_id]
      if match == school:
        affectes[job_id] = affectes[job_id] + 1
        break
  
  proposals = list(map(lambda r : set(matching[r:]) , range(nb_candidats)))
  nb_oui = list(map(lambda school : matching.count(school) , employers))

  #loop for updating popularities  
  popularities = [1] * len(employers)
  while 1:
    popularities, delta = update_popularities(popularities, proposals, nb_oui, employers)
    if delta < 0.01:
      break

  jobs_popularities = []
  jobs_names = []

  #created a sorted list of jobs, ordered by popularity,
  #the most popular first
  sorted_employers_ids = sorted(range(len(employers)), key=lambda x: - popularities[x])
  for employer_id in sorted_employers_ids:
    capacite = capacites[employer_id]
    for _ in range(capacite):
      jobs_names.append(employers[employer_id])
      jobs_popularities.append(popularities[employer_id])

  assert sum(capacites) == len(jobs_names)
  
  return { 
          "nb_candidates" : nb_candidats,
          "employers_names" : employers,
          "employers_popularities" : popularities,
          "employers_capacities" : capacites,
          "jobs_names" : jobs_names,
          "jobs_popularities" : jobs_popularities
        }
  
  
"""one loop of the popularity update"""
def update_popularities(pops, proposals, nb_oui, jobs):
  nb_jobs = len(jobs)
  job_ids = range(nb_jobs)
  
  #precompute inv of sum of popularities of proposals received by each candidate
  candidats = range(len(proposals))
  sum_pops = [0] * len(candidats)
  for job_id in job_ids:
    school = jobs[job_id]
    pop = pops[job_id]
    for c in candidats:
      if school in proposals[c]:
         sum_pops[c] += pop
  sum_inv_pops = list(map(lambda pop : 1.0 / pop, sum_pops))

  new_pops = [0] * len(pops)
  for job_id in job_ids:
    school = jobs[job_id]
    sum_inv = 0.0
    for c in candidats:
      if school in proposals[c]:
        sum_inv += sum_inv_pops[c]
    assert sum_inv > 0
    new_pops[job_id] = nb_oui[job_id] / sum_inv

  #normalize, average pop should be one
  total_pops = sum(new_pops)
  new_pops = list(map(lambda x : nb_jobs * x /total_pops, new_pops))

  delta = map(lambda rank: abs(pops[rank] - new_pops[rank]) / abs(pops[rank] + new_pops[rank]), range(0,len(pops)))

  return new_pops, max(delta)

"""convert a string rank to a numeric rank"""
def to_rank(rang_str):
  if str(rang_str).isnumeric():
    return int(str(rang_str))
  else:
    return -1

"""
Generate a "realistic" popularity profile from data
- some candidates are intrinsically more popular
- jobs have popularities loaded from data
- some candidate-job pairs share interests
We define pop[p,s] = popularity p and s give each other

Pr[p prefers s1 to s2] = pop[p,s1] / (pop[p,s1] + pop[p,s1])
Pr[s prefers p1 to p2] = pop[p1,s] / (pop[p1,s] + pop[p2,s])

Multiplying all popularity by a constant
does not change the distribution

Because we deal with large popularity, we store the log
"""
def generate_logpop(model):
  jobs = model["jobs_names"]
  popularities = model["jobs_popularities"]
  nb_candidates = model["nb_candidates"]

  nb_jobs = len(jobs)
  logpop = np.zeros((nb_jobs, nb_candidates))
  
  # step 1: load jobs pops
  for job_id in range(nb_jobs):
    logpop[job_id,:] += np.log(popularities[job_id])
  
  # step 2: some candidates are intrinsically more popular, but quite uniformly though
  alpha = 0.2
  for s in range(nb_candidates):
    logpop[:,s] += np.log(1/(s+1)**alpha)
  
  # step 3: some candidate-job pairs share interests
  # in that case the mutual popularity is multiplied by the factor
  percent, factor = 0.05, 10
  for _ in range(int(percent*nb_candidates*nb_jobs)):
    p,s = np.random.randint([nb_jobs,nb_candidates])
    logpop[p,s] += np.log(factor)
  
  return logpop

"""save model as csv"""
def serialize(model,filename):
  names = model["employers_names"]
  popularities = model["employers_popularities"]
  capacities = model["employers_capacities"]
  with open(filename + ".csv", "w") as f:
    f.write("nom;capacite;popularite\n")
    min_pop = min(popularities)
    for job_id in range(len(names)):
      school = names[job_id]
      pop = popularities[job_id]
      capa = capacities[job_id]
      f.write('\"' + school[:60] + '\";' + str(capa) + ";" + str(int(pop / min_pop)) + "\n")

  with open(filename + ".json", "w") as f:
    json.dump(model, f)

def deserialize(filename):
  with open(filename + ".json", "r") as f:
    return json.load(f)


def select_wishes(pr, params):

  nb_favorite = params["nb_favorite"]
  nb_medium = params["nb_medium"]
  nb_safe = params["nb_safe"]
  min_rank_medium = params["min_rank_medium"]
  min_rank_safe = params["min_rank_safe"]

  
  # truncate the preference of candidates, (nb_favorite+nb_medium+nb_safe) wishes each
  
  # nb_favorite "favorite" wishes
  result = [pr[i] for i in range(nb_favorite)]

  # nb_medium "medium" wishes, meaning with rank in the popularities order of jobs
  # in the interval [min_rank_medium, min_rank_safe]
  #
  # remark that jobs in the model are sorted by decreasing popularity hence job_id
  # is the rank in the popularity order
  for i in range(nb_favorite + 1,len(pr)):
    job_id = pr[i]
    if not job_id in result and min_rank_medium <= job_id <= min_rank_safe:
      result.append(job_id)
      if(len(result) >= nb_favorite + nb_medium):
        break
  
  # min_rank_safe "secure" wishes with with rank in the popularities order of jobs
  # strictly above min_rank_safe
  for i in range(nb_favorite + 1,len(pr)):
    job_id = pr[i]
    if not job_id in result and min_rank_safe < job_id:
      result.append(job_id)
      if(len(result) >= nb_favorite + nb_medium + nb_safe):
        break

  return result

def select_auditions(job_id, nb_jobs, pref, prefC, params):

  #more auditions for less popular jobs
  #from 8 auditions for the most popular to 16 auditions for the least popular 
  nb_min_auditions = params["nb_min_auditions"] 
  auditions_nb = nb_min_auditions + (nb_min_auditions * job_id) // nb_jobs

  #we add candidates who made a wishes, until the nb of auditions is reached
  result = []
  for i in range(len(pref)):
    st_id = pref[i]
    if job_id in prefC[st_id] :
        result.append(st_id)
    if len(result) >= auditions_nb:
      break

  return result

def truncate(prefJ, prefC, params):
  prefCtrunc = [ select_wishes(pr, params) for pr in prefC ]
  prefJtrunc = []
  for job_id in range(len(prefJ)):
    auditions = select_auditions(job_id, len(prefJ), prefJ[job_id],prefC,params)
    prefJtrunc.append(auditions)
  return prefJtrunc, prefCtrunc

def print_result(model, matchJ, matchC, prefJshort, prefCshort, params):

  print("\n\n****************************************************************")
  print("Paramètres du modèle")

  percent = int(100 * params["nb_voeux"] / params["nb_jobs"])
  print("\tLes candidats choisissent {} voeux (soit {}% du nombre de postes), "
  .format(params["nb_favorite"] + params["nb_medium"]+ params["nb_safe"],percent)
      + "en sélectionnant leurs {} favoris plus {} voeux plus accessibles plus {} voeux de secours."
    .format(params["nb_favorite"], params["nb_medium"], params["nb_safe"])
    )

  print("\tLes employeurs auditionnent pour chaque position entre {} et {} candidats, selon leur popularité."\
    .format(params["nb_min_auditions"] , 2 * params["nb_min_auditions"] ) )

  print("\tA l'issue des classements des candidats auditionnés, les candidats sont affectés en utilisant l'algorithme de Gale-Shapley")
  nb_cand = len(matchC)
  nb_cand_unmatched = matchC.count(None)
  nb_cand_matched = nb_cand - nb_cand_unmatched

  nb_jobs = len(matchJ)
  nb_jobs_unmatched = matchJ.count(None)
  nb_jobs_matched = nb_jobs - nb_jobs_unmatched

  rank_pref_match = [ None if matchC[id] == None else prefCshort[id].index(matchC[id]) for id in range(nb_cand)]
  matched_to_favourite = len(list(filter(lambda rank : rank == 1 ,rank_pref_match)))
  matched_to_favourites = len(list(filter(lambda rank : rank is not None and rank <= 3 ,rank_pref_match)))
  avg_match_rank = np.average(list(filter(lambda x : x is not None, rank_pref_match)))

  rank_pref_match_j = [ None if matchJ[id] == None else prefJshort[id].index(matchJ[id]) for id in range(nb_jobs)]
  matched_to_favourite_j = len(list(filter(lambda rank : rank == 1 ,rank_pref_match_j)))
  matched_to_favourites_j = len(list(filter(lambda rank : rank is not None and rank <= 3 ,rank_pref_match_j)))
  avg_match_rank_j = np.average(list(filter(lambda x : x is not None, rank_pref_match_j)))

  print("Efficacité du premier tour")
  print("\tNombre de candidats avec affectation à l'issue du 1er tour {} / {}.".format(nb_cand_matched , nb_cand))
  print("\tNombre de postes pourvues à l'issue du 1er tour {} / {}.".format(nb_jobs_matched, nb_jobs))
  print("\tNombre de candidats participant au second tour {}.".format(nb_cand_unmatched))
  print("\tNombre de postes proposées au second tour {}.".format(nb_jobs_unmatched))
  print("\n")
  print("Adéquation préférences candidats")
  print("\tNombre de candidats affectés à leur position préférée {}.".format(matched_to_favourite))
  print("\tNombre de candidats affectés à un de leurs trois premières postes préférées {}.".format(matched_to_favourites))
  print("\tRang moyen dans les préférences candidats de la position obtenue {}.".format(int(avg_match_rank)))
  print("\n")
  print("Adéquation préférences formations")
  print("\tNombre de postes recrutant leur candidat préféré {}.".format(matched_to_favourite_j))
  print("\tNombre de postes recrutant un de leurs trois premiers candidats préférés {}.".format(matched_to_favourites_j))
  print("\tRang moyen dans les préférences des postes du candidat recruté {}.".format(int(avg_match_rank_j)))
  
  print("\n")
  print("Liste des employeurs et postes pourvus au premier tour:\n")
  stats = {}
  employers = model["employers_names"]
  pops = model["employers_popularities"]
  for name in employers:
    stats[name] = 0

  for job_id in range(nb_jobs):
    is_matched = (matchJ[job_id] != None)
    if is_matched:
      name = model["jobs_names"][job_id]
      stats[name] = stats[name] + 1
  
  sorted_employers_ids = sorted(range(len(employers)), key=lambda x: - pops[x])
  for employer_id in sorted_employers_ids:
    nb_matched = stats[employers[employer_id]]
    nb_to_match = model["employers_capacities"][employer_id]
    name = employers[employer_id]
    if nb_matched > 0:
      print("'{}' \t: {} / {}.".format(name[:60].ljust(60), nb_matched, nb_to_match))

  print("\n")

  print("Liste des employeurs et postes à pourvoir au second tour:\n")
  sorted_employers_ids = sorted(range(len(employers)), key=lambda x: - pops[x])
  for employer_id in sorted_employers_ids:
    nb_matched = stats[employers[employer_id]]
    nb_to_match = model["employers_capacities"][employer_id] - nb_matched
    name = employers[employer_id]
    if nb_to_match > 0:
      print("'{}' \t: {} position(s) à pourvoir.".format(name[:60].ljust(60), nb_to_match))

def run_experiment(model,params, silent):

  #generates a matrix of log pop 
  logpop = generate_logpop(model)
  
  #draws full preferences
  prefJ, prefC = run.draw_profile(logpop)

  # truncate the preference of jobs
  prefJshort, prefCshort = truncate(prefJ, prefC, params)
      
  # run deferred acceptance
  matchJ, matchC = run.deferred_acceptance(prefJshort, prefCshort)
  
  nb_cand = len(matchC)
  nb_cand_unmatched = matchC.count(None)
  nb_cand_matched = nb_cand - nb_cand_unmatched

  #print the results
  if not silent:
    print_result(model, matchJ, matchC, prefJshort, prefCshort, params)

  return nb_cand_matched

if __name__ == "__main__":
  if len(sys.argv) <= 1 or len(sys.argv) > 2:
    print("Usage: {} [model_filename] ".format(sys.argv[0]), file=sys.stderr)
    model_name = "models/model_medium"
    #sys.exit(0)
  else:
    model_name = sys.argv[1]

  filename = "data/2022_medium.csv"
  try:
    model = deserialize(model_name)
  except Exception as e:
    model = compute_model_from_historique(filename)
    # save to csv and json
    serialize(model, model_name)
  
  
  nb_jobs = len(model["jobs_names"])

  #nombre minimal de voeux tel que fixé dans le projet de décret
  min_nb_voeux = int(np.ceil(0.15 * nb_jobs))

  #nombre minimal d'auditions tel que fixé dans le projet de décret
  min_nb_auditions = 8

  for nb_voeux in (min_nb_voeux, 2 * min_nb_voeux):
    for nb_min_auditions in (min_nb_auditions,2*min_nb_auditions - 1):
      nb_favorite = nb_voeux // 2
      nb_medium = int(np.ceil(nb_voeux / 4))
      nb_safe = nb_voeux - nb_medium - nb_favorite
      params = {
        "nb_jobs" : nb_jobs,
        "min_nb_voeux" : min_nb_voeux,
        "nb_voeux" : nb_voeux,
        "nb_favorite" : nb_favorite,
        "nb_medium" : nb_medium,
        "nb_safe" : nb_safe,
        "min_rank_medium" : nb_favorite  + 1,
        "min_rank_safe" :  1 + nb_jobs // 2,
        "nb_min_auditions" : nb_min_auditions
      }
      nb_matched_stats = []
      for i in range(500):
        silent = (i != 0)
        nb_matched_stats.append(run_experiment(model, params, silent))
      nb_candidates = model["nb_candidates"]
      avg_matched = int(np.average(nb_matched_stats))
      min_matched = np.min(nb_matched_stats)
      max_matched = np.max(nb_matched_stats)
      print("\n******\nnb_voeux {} nb_min_auditions {} ".format(nb_voeux, nb_min_auditions), file=sys.stderr)
      print("[min,avg,max] matched [{},{},{}]/{}\n".format(min_matched, avg_matched, max_matched, nb_candidates), file=sys.stderr)


