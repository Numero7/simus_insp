import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


"""
In each scenario:
- we let students apply to their top X choices
- we let positions keep their top Y applicants
"""
nbPositions, nbStudents = 85, 80
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
  alpha = 3
  for s in range(nbStudents):
    logpop[:,s] += np.log(1/(s+1)**alpha)
  
  # step 3: some student-position pairs share interests
  percent, factor = 0.10, 10
  for _ in range(int(percent*nbStudents*nbPositions)):
    p,s = np.random.randint([nbPositions,nbStudents])
    logpop[p,s] += np.log(factor)
  
  return logpop
  
"""
Incertitude on an average
  we display "avg +- incertitude * std / sqrt(nbRuns)"
where:
  avg = empirical average
  std = empirical standard deviation
"""
nbRuns = 100
incertitude = 2 # 95% for gaussian random variables

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
  return sorted(range(n), key=lambda i:r[i]-logpop[i])

def draw_profile(nbPositions, nbStudents, seed=None):
  if seed is not None:
    np.random.seed(seed)
  
  # build popularity profile
  logpop = generate_logpop(nbPositions, nbStudents)
  
  # draw preferences
  nbPositions, nbStudents = logpop.shape
  prefP = [draw_pref(logpop[p,:]) for p in range(nbPositions)]
  prefS = [draw_pref(logpop[:,s]) for s in range(nbStudents)]
  return logpop, prefP, prefS

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

def plot_run(filename, logpop, prefs):
  nbPositions, nbStudents = logpop.shape
  
  with PdfPages(filename) as pdf:
    
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(
      vmin=logpop.min(), vmax=logpop.max())
    txt1 = "First, students apply to %d positions" % sz1
    txt2 = "Then, positions keep at most %d applicants" % sz2
    
    #####################################
    
    plt.figure(figsize=(6, 5), tight_layout=True)
    
    plt.imshow(logpop, origin='lower', norm=norm, cmap=cmap)
    plt.xlabel("id student")
    plt.ylabel("id position")
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
    prefP, prefS = prefs[1]
    
    tmp = logpop.copy()
    for p in range(nbPositions):
      l = set(range(nbStudents)) - set(prefP[p])
      for s in l:
        tmp[p,s] = None
      if prefP[p] == []:
        plt.axhline(p, color="r")
        
    plt.imshow(tmp, origin='lower', norm=norm, cmap=cmap)
    plt.xlabel("id student")
    plt.ylabel("id position")
    
    ax = plt.subplot(1,2,2)
    ax.title.set_text(txt2)
    prefP, prefS = prefs[2]
    
    for s in range(nbStudents):
      l = set(range(nbPositions)) - set(prefS[s])
      for p in l:
        tmp[p,s] = None
      if prefS[s] == []:
        plt.axvline(s, color="r")
    
    plt.imshow(tmp, origin='lower', norm=norm, cmap=cmap)
    plt.xlabel("id student")
    plt.ylabel("id position")
    
    pdf.savefig()
    plt.close()
    
    #####################################
    plt.figure(figsize=(10, 3), tight_layout=True)
    
    ax = plt.subplot(1,2,1)
    ax.title.set_text(txt1)
    prefP, prefS = prefs[1]
    
    YS = sorted([len(pr) for pr in prefS], reverse=True)
    YP = sorted([len(pr) for pr in prefP], reverse=True)
    plt.plot(YS, label="length of students' lists")
    plt.plot(YP, label="length of positions' lists")
    plt.ylim(bottom=0)
    plt.legend()
    
    ax = plt.subplot(1,2,2)
    ax.title.set_text(txt2)
    prefP, prefS = prefs[2]
    
    YS = sorted([len(pr) for pr in prefS], reverse=True)
    YP = sorted([len(pr) for pr in prefP], reverse=True)
    plt.plot(YS, label="length of students' lists")
    plt.plot(YP, label="length of positions' lists")
    plt.ylim(bottom=0)
    plt.legend()
    
    pdf.savefig()
    plt.close()


def plot_aggregated(filename, nbMatchs, nbInterviews):
  global scenarios, nbRuns
  nbScenarios, nbPositions, nbStudents = nbMatchs.shape
  
  with PdfPages(filename) as pdf:
    
    cmap = mpl.cm.viridis
    norm = mpl.colors.LogNorm(vmin=1/nbRuns, vmax=1)
    
    #####################################
    plt.figure(figsize=(10, 4), tight_layout=True)
    
    ax = plt.subplot(1,2,1)
    for iScenario, (name, sz1, sz2) in enumerate(scenarios):
      color = plt.cm.tab10(iScenario)
      
      avg = nbInterviews[iScenario,:,:].sum(axis=1) / nbRuns
      #avg = sum(lenP[iScenario,:,i]*i
      #  for i in range(nbStudents+1)) / nbRuns
      #var = sum(lenP[iScenario,:,i]*(i-avg)**2
      #  for i in range(nbStudents+1)) / nbRuns
      #err = var**.5 * incertitude / nbRuns**.5
      
      plt.plot(avg, ".-", color=color, label=name)
      #plt.fill_between(range(nbPositions), avg-err, avg+err,
      #  color=color, alpha=.1)
    
    plt.ylabel("average number of interviews")
    plt.xlabel("id position")
    plt.legend()
    
    ax = plt.subplot(1,2,2)
    for iScenario, (name, sz1, sz2) in enumerate(scenarios):
      color = plt.cm.tab10(iScenario)
      
      avg = nbInterviews[iScenario,:,:].sum(axis=0) / nbRuns
      #avg = sum(lenS[iScenario,:,i]*i
      #  for i in range(nbPositions+1)) / nbRuns
      #var = sum(lenS[iScenario,:,i]*(i-avg)**2
      #  for i in range(nbPositions+1)) / nbRuns
      #err = var**.5 * incertitude / nbRuns**.5
      
      plt.plot(avg, ".-", color=color, label=name)
      #plt.fill_between(range(nbStudents), avg-err, avg+err,
      #  color=color, alpha=.1)
    
    plt.ylabel("average number of interviews")
    plt.xlabel("id student")
    plt.legend()
    
    pdf.savefig()
    plt.close()
    
    #####################################
    plt.figure(figsize=(10, 4), tight_layout=True)
    
    ax = plt.subplot(1,2,1)
    for iScenario, (name, sz1, sz2) in enumerate(scenarios):
      color = plt.cm.tab10(iScenario)
      
      pr = 1 - nbMatchs[iScenario,:,:].sum(axis=1) / nbRuns
      std = (2*pr*(1-pr))**.5
      err = std*incertitude/nbRuns**.5
      
      plt.plot(pr, ".-", color=color, label=name)
      plt.fill_between(range(nbPositions), pr-err, pr+err,
        color=color, alpha=.1)
    
    plt.ylabel("probability of being unassigned")
    plt.xlabel("id position")
    plt.ylim((0,1))
    plt.legend()
    
    ax = plt.subplot(1,2,2)
    for iScenario, (name, sz1, sz2) in enumerate(scenarios):
      color = plt.cm.tab10(iScenario)
      
      pr = 1 - nbMatchs[iScenario,:,:].sum(axis=0) / nbRuns
      std = (2*pr*(1-pr))**.5
      err = std*incertitude/nbRuns**.5
      
      plt.plot(pr, ".-", color=color, label=name)
      plt.fill_between(range(nbStudents), pr-err, pr+err,
        color=color, alpha=.1)
    
    plt.ylabel("probability of being unassigned")
    plt.xlabel("id student")
    plt.ylim((0,1))
    plt.legend()
    
    pdf.savefig()
    plt.close()
    
    #####################################
    for iScenario, (name, sz1, sz2) in enumerate(scenarios):
      plt.figure(figsize=(10, 4), tight_layout=True)
      plt.title(name)
      
      ax = plt.subplot(1,2,1)
      ax.title.set_text("Interview")
      plt.imshow(nbInterviews[iScenario]/nbRuns,
        origin='lower', norm=norm, cmap=cmap)
      plt.xlabel("id student")
      plt.ylabel("id position")
      plt.colorbar()
        
      ax = plt.subplot(1,2,2)
      ax.title.set_text("Match")
      plt.imshow(nbMatchs[iScenario]/nbRuns,
        origin='lower', norm=norm, cmap=cmap)
      plt.xlabel("id student")
      plt.ylabel("id position")
      plt.colorbar()
      
      pdf.savefig()
      plt.close()
  

if __name__ == "__main__":

  # aggregated data
  nbScenarios = len(scenarios)
  #lenP = np.zeros((nbScenarios, nbPositions, nbStudents+1))
  #lenS = np.zeros((nbScenarios, nbStudents, nbPositions+1,))
  nbMatchs = np.zeros((nbScenarios, nbPositions, nbStudents))
  nbInterviews = np.zeros((nbScenarios, nbPositions, nbStudents))
    
  for iScenario, (name, sz1, sz2) in enumerate(scenarios):
    print("Running", name, "...")

    for iRun in range(nbRuns):
      print("\r", iRun+1, "/", nbRuns, end="", flush=True)
      
      # draw random preferences
      logpop, prefP, prefS = draw_profile(
        nbPositions, nbStudents, iRun)
      prefs = [(prefP, prefS)]
      
      # truncate the preference of students
      prefS, prefP = truncate(prefS, prefP, sz1)
      prefs.append((prefP, prefS))
      
      # truncate the preference of positions
      prefP, prefS = truncate(prefP, prefS, sz2)
      prefs.append((prefP, prefS))
      
      # run deferred acceptance
      matchP, matchS = deferred_acceptance(prefP, prefS)
      
      # aggregated data
      for iPosition in range(nbPositions):
        for iStudent in prefP[iPosition]:
          nbInterviews[iScenario,iPosition, iStudent] += 1
        if matchP[iPosition] != None:
          nbMatchs[iScenario,iPosition,matchP[iPosition]] += 1
      #for iPosition, pr in enumerate(prefP):
      #  lenP[iScenario,iPosition,len(pr)] += 1
      #for iStudent, pr in enumerate(prefS):
      #  lenS[iScenario,iStudent,len(pr)] += 1
    
      # plot this run
      if iRun == 0: 
        name = "fig-scenario-%d-run-%d.pdf" % (iScenario+1, iRun+1)
        plot_run(name, logpop, prefs)
    
    print(" done!")
  
  plot_aggregated("fig.pdf", nbMatchs, nbInterviews)
    
  
