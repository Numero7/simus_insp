import pandas as pd
import json as json
import sys as sys
import numpy as np

"""
load a realistic popularity profile from 2022 data
"""


def compute_model_from_historique(filename):
    df = pd.read_csv(filename, sep=";")
    df["name"] = df["Ministère"] + df["Poste"]

    # create lists from panda format
    matching = df["Ministère"].tolist()

    employers = list(set(matching))
    capacites = list(map(lambda school: matching.count(school), employers))

    nb_employers = len(employers)
    affectes = [0] * nb_employers
    nb_candidats = sum(map(lambda r: str(r).isnumeric(), df["Rang"].tolist()))
    for rang in range(nb_candidats):
        match = matching[rang]
        for job_id in range(nb_employers):
            school = employers[job_id]
            if match == school:
                affectes[job_id] = affectes[job_id] + 1
                break

    proposals = list(map(lambda r: set(matching[r:]), range(nb_candidats)))
    nb_oui = list(map(lambda school: matching.count(school), employers))

    # loop for updating popularities
    popularities = [1] * len(employers)
    while 1:
        popularities, delta = update_popularities(
            popularities, proposals, nb_oui, employers
        )
        if delta < 0.01:
            break

    jobs_popularities = []
    jobs_names = []

    # created a sorted list of jobs, ordered by popularity,
    # the most popular first
    sorted_employers_ids = sorted(range(len(employers)), key=lambda x: -popularities[x])
    for employer_id in sorted_employers_ids:
        capacite = capacites[employer_id]
        for _ in range(capacite):
            jobs_names.append(employers[employer_id])
            jobs_popularities.append(popularities[employer_id])

    assert sum(capacites) == len(jobs_names)

    return {
        "nb_candidates": nb_candidats,
        "employers_names": employers,
        "employers_popularities": popularities,
        "employers_capacities": capacites,
        "jobs_names": jobs_names,
        "jobs_popularities": jobs_popularities,
    }


"""one loop of the popularity update"""


def update_popularities(pops, proposals, nb_oui, jobs):
    nb_jobs = len(jobs)
    job_ids = range(nb_jobs)

    # precompute inv of sum of popularities of proposals received by each candidate
    candidats = range(len(proposals))
    sum_pops = [0] * len(candidats)
    for job_id in job_ids:
        school = jobs[job_id]
        pop = pops[job_id]
        for c in candidats:
            if school in proposals[c]:
                sum_pops[c] += pop
    sum_inv_pops = list(map(lambda pop: 1.0 / pop, sum_pops))

    new_pops = [0] * len(pops)
    for job_id in job_ids:
        school = jobs[job_id]
        sum_inv = 0.0
        for c in candidats:
            if school in proposals[c]:
                sum_inv += sum_inv_pops[c]
        assert sum_inv > 0
        new_pops[job_id] = nb_oui[job_id] / sum_inv

    # normalize, average pop should be one
    total_pops = sum(new_pops)
    new_pops = list(map(lambda x: nb_jobs * x / total_pops, new_pops))

    delta = map(
        lambda rank: abs(pops[rank] - new_pops[rank])
        / abs(pops[rank] + new_pops[rank]),
        range(0, len(pops)),
    )

    return new_pops, max(delta)


"""save model as csv"""


def serialize(model, filename):
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
            f.write(
                '"'
                + school[:60]
                + '";'
                + str(capa)
                + ";"
                + str(int(pop / min_pop))
                + "\n"
            )

    with open(filename + ".json", "w") as f:
        json.dump(model, f)


def deserialize(filename):
    with open(filename + ".json", "r") as f:
        return json.load(f)


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


def generate_logpop_from_model(model):
    jobs = model["jobs_names"]
    popularities = model["jobs_popularities"]
    nb_candidates = model["nb_candidates"]

    nb_jobs = len(jobs)
    logpop = np.zeros((nb_jobs, nb_candidates))

    # step 1: load jobs pops
    for job_id in range(nb_jobs):
        logpop[job_id, :] += np.log(popularities[job_id])

    # step 2: some candidates are intrinsically more popular, but quite uniformly though
    alpha = 0.2
    for s in range(nb_candidates):
        logpop[:, s] += np.log(1 / (s + 1) ** alpha)

    # step 3: some candidate-job pairs share interests
    # in that case the mutual popularity is multiplied by the factor
    percent, factor = 0.05, 10
    for _ in range(int(percent * nb_candidates * nb_jobs)):
        p, s = np.random.randint([nb_jobs, nb_candidates])
        logpop[p, s] += np.log(factor)

    return logpop


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
        logpop[p, :] += np.log(1 / (p + 1) ** alpha)

    # step 2: some positions are intrinsically more popular
    alpha = 2
    for s in range(nbStudents):
        logpop[:, s] += np.log(1 / (s + 1) ** alpha)

    # step 3: some student-position pairs share interests
    percent, factor = 0.05, 10
    for _ in range(int(percent * nbStudents * nbPositions)):
        p, s = np.random.randint([nbPositions, nbStudents])
        logpop[p, s] += np.log(factor)

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
    result = sorted(range(n), key=lambda i: r[i] - logpop[i])
    return result


def draw_profile(logpop):
    nbPositions, nbStudents = logpop.shape
    prefP = [draw_pref(logpop[p, :]) for p in range(nbPositions)]
    prefS = [draw_pref(logpop[:, s]) for s in range(nbStudents)]
    return prefP, prefS


if __name__ == "__main__":
    filename = "data/2022_medium.csv"
    model_name = "models/model_medium"
    if len(sys.argv) <= 1:
        print(
            "Using defaults: input file " + filename + " and output file " + model_name
        )
    elif len(sys.argv) == 1:
        model_name = sys.argv[1]
        print("Using : input file " + filename + " and output file " + model_name)
    else:
        print("Usage: {} [model_filename] ".format(sys.argv[0]), file=sys.stderr)
        # sys.exit(0)

    print("Computing model...")
    popularities = compute_model_from_historique(filename)
    # save to csv and json

    print("Saving model to file " + model_name + "...")
    serialize(popularities, model_name)

    print("Done.")
