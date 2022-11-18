import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

nbPositions, nbStudents = 85, 80

"""
In each scenario:
- we let students apply to their top X choices
- we let positions keep their top Y applicants
"""
scenarios = [
  ("Scenario 1", int(0.15 * nbPositions), 8),
  ("Scenario 2", int(0.15 * nbPositions), nbStudents),
  ("Scenario 3", nbPositions,             8),
  ("Scenario 4", nbPositions,             nbStudents),
]

"""
Generate a "realistic" popularity profile
- some students are intrinsically more popular
- some positions are intrinsically more popular
- some student-position pairs share interests
We define pop[p,s] = popularity p and s give each other

Pr[p prefers s1 to s2] = pop[p,s1] / (pop[p,s1] + pop[p,s1])
Pr[s prefers p1 to p2] = pop[p1,s] / (pop[p1,s] + pop[p2,s])

Multiplying all popularity by a constant
does not change the distribution

Because we deal with large popularity, we store the log
"""
def generate_logpop(nbPositions, nbStudents):
  logpop = np.zeros((nbPositions, nbStudents))
  
  # step 1: some students are intrinsically more popular
  alpha = 1
  for p in range(nbPositions):
    logpop[p,:] += np.log(1/(p+1)**alpha)
  
  # step 2: some positions are intrinsically more popular
  alpha = 2
  for s in range(nbStudents):
    logpop[:,s] += np.log(1/(s+1)**alpha)
  
  # step 3: some student-position pairs share interests
  percent, factor = 0.05, 10
  for _ in range(int(percent*nbStudents*nbPositions)):
    p,s = np.random.randint([nbPositions,nbStudents])
    logpop[p,s] += np.log(factor)
  
  return logpop

##### IT SHOULD NOT BE NECESSARY TO CHANGE THINGS BELOW #####

"""
Recall that we want a distribution such that
Pr[a > b] = pop[a] / (pop[a] + pop[b])

We draw without replacement with proba proportional to pop
Pr[a > b > ... > z] = pop[a] / (pop[a]+pop[b]+...+pop[z])
                    * pop[b] / (pop[b]+...+pop[z])
                    * ...
                    * pop[z] / (pop[z])
 <=> sort by increasing X[i] drawn from Exp(pop[i])
 <=> sort by increasing X[i] = -log(Unif)/pop[i]
 <=> sort by increasing Y[i] = log(-log(Unif))-log(pop[i])
"""

def draw_pref(logpop):
  n = len(logpop)
  r = np.log(-np.log(np.random.rand(n)))
  result = sorted(range(n), key=lambda i:r[i]-logpop[i])
  return result

def draw_profile(logpop):
  nbPositions, nbStudents = logpop.shape
  prefP = [draw_pref(logpop[p,:]) for p in range(nbPositions)]
  prefS = [draw_pref(logpop[:,s]) for s in range(nbStudents)]
  return prefP, prefS

"""
administration-proposing deferred acceptance
takes as input (possibly incomplete) preference lists
"""
def deferred_acceptance(prefP, prefS):
  nbPositions, nbStudents = len(prefP), len(prefS)
  
  # rankS[s][p] = rank of p in the list of s (None is last)
  rankS = [{p:r for r,p in enumerate(pr+[None])} for pr in prefS]
  
  # propP[p] = rank of the next proposal from p
  propP = [0] * nbPositions
  
  # matchS[s] = tentative match of s
  matchS = [None] * nbStudents
  
  p = 0
  while p < nbPositions:
    s = None
    if propP[p] < len(prefP[p]):
      s = prefP[p][propP[p]]
    if s != None and matchS[s] != p:
      if p in rankS[s] and rankS[s][p] < rankS[s][matchS[s]]:
        next = p+1
        if matchS[s] != None:                
          propP[matchS[s]] += 1
          next = matchS[s]
        matchS[s] = p
        p = next
      else:
        propP[p] += 1
    else:
      p += 1
  
  # matchP[p] = match of p
  matchP = [None] * nbPositions
  for s,p in enumerate(matchS):
    if p != None:
      matchP[p] = s
  
  return matchP, matchS

def truncate(prefA, prefB, sz):
  prA = [pr[:sz] for pr in prefA]
  prB = [[i for i in pr if j in prA[i]] # todo: not efficient
    for j,pr in enumerate(prefB)]
  return prA, prB

if __name__ == "__main__":

  logpop = generate_logpop(nbPositions, nbStudents)
  prefP1, prefS1 = draw_profile(logpop)
  
  with PdfPages('fig.pdf') as pdf:
    for iScenario, (name, sz1, sz2) in enumerate(scenarios):
    
      # truncate the preference of students
      prefS2, prefP2 = truncate(prefS1, prefP1, sz1)
      
      # truncate the preference of positions
      prefP3, prefS3 = truncate(prefP2, prefS2, sz2)
      
      # run deferred acceptance
      matchP, matchS = deferred_acceptance(prefP3, prefS3)
    
      #####################################
    
      cmap = mpl.cm.viridis
      norm = mpl.colors.Normalize(
        vmin=logpop.min(), vmax=logpop.max())
      txt1 = "First, students apply to %d positions" % sz1
      txt2 = "Then, positions keep at most %d applicants" % sz2
      
      #####################################
      
      plt.figure(figsize=(6, 5), tight_layout=True)
      plt.title(name)
      
      plt.imshow(logpop, origin='lower', norm=norm, cmap=cmap)
      plt.xlabel("students")
      plt.ylabel("positions")
      plt.colorbar()
      
      for s in range(nbStudents):
        if matchS[s] == None:
          plt.plot([s],[nbPositions], "r.", markersize=3)
        else:
          plt.plot([s],[matchS[s]], "r.", markersize=3)
      for p in range(nbPositions):
        if matchP[p] == None:
          plt.plot([nbStudents],[p], "r.", markersize=3)
      
      plt.xlim((-.5,nbStudents+.5))
      plt.ylim((-.5,nbPositions+.5))
      
      pdf.savefig()
      plt.close()
      
      #####################################
      plt.figure(figsize=(10, 5), tight_layout=True)
      
      ax = plt.subplot(1,2,1)
      ax.title.set_text(txt1)
      
      tmp = logpop.copy()
      for p in range(nbPositions):
        l = set(range(nbStudents)) - set(prefP2[p])
        for s in l:
          tmp[p,s] = None
        if prefP2[p] == []:
          plt.axhline(p, color="r")
          
      plt.imshow(tmp, origin='lower', norm=norm, cmap=cmap)
      plt.xlabel("students")
      plt.ylabel("positions")
      
      ax = plt.subplot(1,2,2)
      ax.title.set_text(txt2)
      
      for s in range(nbStudents):
        l = set(range(nbPositions)) - set(prefS3[s])
        for p in l:
          tmp[p,s] = None
        if prefS3[s] == []:
          plt.axvline(s, color="r")
      
      plt.imshow(tmp, origin='lower', norm=norm, cmap=cmap)
      plt.xlabel("students")
      plt.ylabel("positions")
      
      pdf.savefig()
      plt.close()
      
      #####################################
      plt.figure(figsize=(10, 3), tight_layout=True)
      
      ax = plt.subplot(1,2,1)
      ax.title.set_text(txt1)
      
      YS = sorted([len(pr) for pr in prefS2], reverse=True)
      YP = sorted([len(pr) for pr in prefP2], reverse=True)
      plt.plot(YS, label="length of students' lists")
      plt.plot(YP, label="length of positions' lists")
      plt.ylim(bottom=0)
      plt.legend()
      
      ax = plt.subplot(1,2,2)
      ax.title.set_text(txt2)
      
      YS = sorted([len(pr) for pr in prefS3], reverse=True)
      YP = sorted([len(pr) for pr in prefP3], reverse=True)
      plt.plot(YS, label="length of students' lists")
      plt.plot(YP, label="length of positions' lists")
      plt.ylim(bottom=0)
      plt.legend()
      
      pdf.savefig()
      plt.close()
  
