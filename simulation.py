import pandas as pd
import numpy as np
import random as rnd
import json as json
import sys as sys
import da
import popularities as pop

"""convert a string rank to a numeric rank"""


def to_rank(rang_str):
    if str(rang_str).isnumeric():
        return int(str(rang_str))
    else:
        return -1


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
    for i in range(nb_favorite + 1, len(pr)):
        job_id = pr[i]
        if not job_id in result and min_rank_medium <= job_id <= min_rank_safe:
            result.append(job_id)
            if len(result) >= nb_favorite + nb_medium:
                break

    # min_rank_safe "secure" wishes with with rank in the popularities order of jobs
    # strictly above min_rank_safe
    for i in range(nb_favorite + 1, len(pr)):
        job_id = pr[i]
        if not job_id in result and min_rank_safe < job_id:
            result.append(job_id)
            if len(result) >= nb_favorite + nb_medium + nb_safe:
                break

    return result


def select_auditions(job_id, nb_jobs, pref, prefC, params):
    # more auditions for less popular jobs
    # from 8 auditions for the most popular to 16 auditions for the least popular
    nb_min_auditions = params["nb_min_auditions"]
    auditions_nb = nb_min_auditions + (nb_min_auditions * job_id) // nb_jobs

    # we add candidates who made a wishes, until the nb of auditions is reached
    result = []
    for i in range(len(pref)):
        st_id = pref[i]
        if job_id in prefC[st_id]:
            result.append(st_id)
        if len(result) >= auditions_nb:
            break

    return result


def truncate(prefJ, prefC, params):
    prefCtrunc = [select_wishes(pr, params) for pr in prefC]
    prefJtrunc = []
    for job_id in range(len(prefJ)):
        auditions = select_auditions(job_id, len(prefJ), prefJ[job_id], prefC, params)
        prefJtrunc.append(auditions)
    return prefJtrunc, prefCtrunc


def print_result(model, matchJ, matchC, prefJshort, prefCshort, params, nb_dev):
    print("\n\n****************************************************************")
    print("Paramètres du modèle")

    percent = int(100 * params["nb_voeux"] / params["nb_jobs"])
    print(
        "\tLes candidats choisissent {} voeux (soit {}% du nombre de postes), ".format(
            params["nb_favorite"] + params["nb_medium"] + params["nb_safe"], percent
        )
        + "en sélectionnant leurs {} favoris plus {} voeux plus accessibles plus {} voeux de secours.".format(
            params["nb_favorite"], params["nb_medium"], params["nb_safe"]
        )
    )

    print(
        "\tLes employeurs auditionnent pour chaque position entre {} et {} candidats, selon leur popularité.".format(
            params["nb_min_auditions"], 2 * params["nb_min_auditions"]
        )
    )

    print(
        "\tA l'issue des classements des candidats auditionnés, les candidats sont affectés en utilisant l'algorithme de Gale-Shapley"
    )
    nb_cand = len(matchC)
    nb_cand_unmatched = matchC.count(None)
    nb_cand_matched = nb_cand - nb_cand_unmatched

    nb_jobs = len(matchJ)
    nb_jobs_unmatched = matchJ.count(None)
    nb_jobs_matched = nb_jobs - nb_jobs_unmatched

    rank_pref_match = [
        None if matchC[id] == None else prefCshort[id].index(matchC[id])
        for id in range(nb_cand)
    ]

    matched_to_favourite = len(
        list(filter(lambda rank: rank is not None and rank == 0, rank_pref_match))
    )
    matched_to_second = len(
        list(filter(lambda rank: rank is not None and rank == 1, rank_pref_match))
    )
    matched_to_third = len(
        list(filter(lambda rank: rank is not None and rank == 2, rank_pref_match))
    )
    matched_to_favourites = len(
        list(filter(lambda rank: rank is not None and rank <= 2, rank_pref_match))
    )

    avg_match_rank = 1 + np.average(
        list(filter(lambda x: x is not None, rank_pref_match))
    )

    rank_pref_match_j = [
        None if matchJ[id] == None else prefJshort[id].index(matchJ[id])
        for id in range(nb_jobs)
    ]

    matched_to_favourite_j = len(
        list(filter(lambda rank: rank == 1, rank_pref_match_j))
    )
    matched_to_favourites_j = len(
        list(filter(lambda rank: rank is not None and rank <= 3, rank_pref_match_j))
    )
    avg_match_rank_j = np.average(
        list(filter(lambda x: x is not None, rank_pref_match_j))
    )

    print("\n")
    print("Adéquation préférences formations")
    print(
        "\tNombre de postes recrutant leur candidat préféré {}.".format(
            matched_to_favourite_j
        )
    )
    print(
        "\tNombre de postes recrutant un de leurs trois premiers candidats préférés {}.".format(
            matched_to_favourites_j
        )
    )
    print(
        "\tRang moyen dans les préférences des postes du candidat recruté {}.".format(
            int(avg_match_rank_j)
        )
    )

    print("\n")
    print("Liste des employeurs et postes pourvus au premier tour:\n")
    stats = {}
    employers = model["employers_names"]
    pops = model["employers_popularities"]
    for name in employers:
        stats[name] = 0

    for job_id in range(nb_jobs):
        is_matched = matchJ[job_id] != None
        if is_matched:
            name = model["jobs_names"][job_id]
            stats[name] = stats[name] + 1

    sorted_employers_ids = sorted(range(len(employers)), key=lambda x: -pops[x])
    for employer_id in sorted_employers_ids:
        nb_matched = stats[employers[employer_id]]
        nb_to_match = model["employers_capacities"][employer_id]
        name = employers[employer_id]
        if nb_matched > 0:
            print(
                "'{}' \t: {} / {}.".format(name[:60].ljust(60), nb_matched, nb_to_match)
            )

    print("\n")

    print("Liste des employeurs et postes à pourvoir au second tour:\n")
    sorted_employers_ids = sorted(range(len(employers)), key=lambda x: -pops[x])
    for employer_id in sorted_employers_ids:
        nb_matched = stats[employers[employer_id]]
        nb_to_match = model["employers_capacities"][employer_id] - nb_matched
        name = employers[employer_id]
        if nb_to_match > 0:
            print(
                "'{}' \t: {} position(s) à pourvoir.".format(
                    name[:60].ljust(60), nb_to_match
                )
            )

    print("Efficacité du premier tour")
    print(
        "\tNombre de candidats avec affectation à l'issue du 1er tour {} / {}.".format(
            nb_cand_matched, nb_cand
        )
    )
    print(
        "\tNombre de postes pourvues à l'issue du 1er tour {} / {}.".format(
            nb_jobs_matched, nb_jobs
        )
    )
    print(
        "\tNombre de candidats participant au second tour {}.".format(nb_cand_unmatched)
    )
    print("\tNombre de postes proposées au second tour {}.".format(nb_jobs_unmatched))
    print("\n")
    print("Adéquation préférences candidats")
    print(
        "\tNombre de candidats affectés à leur poste préféré {}.".format(
            matched_to_favourite
        )
    )
    print(
        "\tNombre de candidats affectés à leur second poste préféré {}.".format(
            matched_to_second
        )
    )
    print(
        "\tNombre de candidats affectés à leur troisième poste préféré {}.".format(
            matched_to_third
        )
    )
    print(
        "\tNombre de candidats affectés à un de leurs trois premiers postes préférés {}.".format(
            matched_to_favourites
        )
    )
    print(
        "\tRang moyen dans les préférences candidats de la position obtenue {}.".format(
            int(avg_match_rank)
        )
    )
    rangs = list(
        map(lambda rank: 1 + rank if rank is not None else "", rank_pref_match)
    )

    print(
        "\t Nombre de candidats matchés au premier tour et ayant une incitation à tenter leur chance au second tour {}/{}.".format(
            nb_dev, nb_cand
        )
    )

    print(
        "\tListe des rangs du matching au premier tour dans les listes de préférence des candidats {}.".format(
            rangs
        )
    )


def run_experiment(model, params, silent):
    # generates a matrix of log pop
    logpop = pop.generate_logpop_from_model(model)

    # draws full preferences
    prefJ, prefC = pop.draw_profile(logpop)

    # truncate the preference of jobs
    prefJshort, prefCshort = truncate(prefJ, prefC, params)

    # run deferred acceptance
    matchJ, matchC = da.deferred_acceptance(prefJshort, prefCshort)

    nb_cand = len(matchC)
    nb_cand_unmatched = matchC.count(None)
    nb_cand_matched = nb_cand - nb_cand_unmatched

    rank_pref_match = [
        None if matchC[id] == None else prefCshort[id].index(matchC[id])
        for id in range(nb_cand)
    ]

    nb_dev = 0
    is_matched = list(map(lambda r: 1 if r is not None else 0, matchJ))
    for id in range(nb_cand):
        r = rank_pref_match[id]
        if r is not None:
            for rank in range(r):
                job_preferred = prefC[id][rank]
                if is_matched[job_preferred] == 0:
                    nb_dev = nb_dev + 1
                    break

    # print the results
    if not silent:
        print_result(model, matchJ, matchC, prefJshort, prefCshort, params, nb_dev)

    return nb_cand_matched, nb_dev


if __name__ == "__main__":
    if len(sys.argv) <= 1 or len(sys.argv) > 2:
        print("Usage: {} [model_filename] ".format(sys.argv[0]), file=sys.stderr)
        model_name = "models/model_medium"
        # sys.exit(0)
    else:
        model_name = sys.argv[1]

    popularities = pop.deserialize(model_name)

    nb_jobs = len(popularities["jobs_names"])

    # nombre minimal de voeux tel que fixé dans le projet de décret
    min_nb_voeux = int(np.ceil(0.15 * nb_jobs))

    # nombre minimal d'auditions tel que fixé dans le projet de décret
    min_nb_auditions = 8

    # for nb_voeux in (min_nb_voeux, 2 * min_nb_voeux):
    #    for nb_min_auditions in (min_nb_auditions, 2 * min_nb_auditions - 1):
    for nb_voeux in [min_nb_voeux]:
        for nb_min_auditions in [min_nb_auditions]:
            nb_favorite = nb_voeux // 2
            nb_medium = int(np.ceil(nb_voeux / 2))
            nb_safe = 0  # nb_voeux - nb_medium - nb_favorite
            params = {
                "nb_jobs": nb_jobs,
                "min_nb_voeux": min_nb_voeux,
                "nb_voeux": nb_voeux,
                "nb_favorite": nb_favorite,
                "nb_medium": nb_medium,
                "nb_safe": nb_safe,
                "min_rank_medium": nb_favorite + 1,
                "min_rank_safe": 1 + nb_jobs // 2,
                "nb_min_auditions": nb_min_auditions,
            }
            nb_matched_stats = []
            nb_dev_stats = []
            for i in range(500):
                silent = i != 0
                nb_matched, nb_dev = run_experiment(popularities, params, silent)
                nb_matched_stats.append(nb_matched)
                nb_dev_stats.append(nb_dev)
            nb_candidates = popularities["nb_candidates"]
            avg_matched = int(np.average(nb_matched_stats))
            min_matched = np.min(nb_matched_stats)
            max_matched = np.max(nb_matched_stats)
            avg_dev = int(np.average(nb_dev_stats))
            min_dev = np.min(nb_dev_stats)
            max_dev = np.max(nb_dev_stats)
            print(
                "\n******\nnb_voeux {} nb_min_auditions {} ".format(
                    nb_voeux, nb_min_auditions
                ),
                file=sys.stderr,
            )
            print(
                "Statistics on 500 experiments [min,avg,max] matched after first step [{},{},{}] for {} candidates\n".format(
                    min_matched, avg_matched, max_matched, nb_candidates
                ),
                file=sys.stderr,
            )
            print(
                "Statistics on 500 experiments [min,avg,max] candidates matched at first run wanting to participate again to the second run [{},{},{}] for {} candidates\n".format(
                    min_dev, avg_dev, max_dev, nb_candidates
                ),
                file=sys.stderr,
            )
