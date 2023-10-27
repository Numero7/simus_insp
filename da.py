"""
administration-proposing deferred acceptance
takes as input (possibly incomplete) preference lists
"""


def deferred_acceptance(prefP, prefS):
    nbPositions, nbStudents = len(prefP), len(prefS)

    # rankS[s][p] = rank of p in the list of s (None is last)
    rankS = [{p: r for r, p in enumerate(pr + [None])} for pr in prefS]

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
                next = p + 1
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
    for s, p in enumerate(matchS):
        if p != None:
            matchP[p] = s

    return matchP, matchS
