import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import popularities as pop
from da import deferred_acceptance
from matplotlib.backends.backend_pdf import PdfPages

"""npPositions is the number of open positions in Ministries"""
nbPositions, nbStudents = 85, 80

"""
In each scenario:
- we let students apply to their top X choices
- we let positions keep their top Y applicants
"""
scenarios = [
    ("Scenario 1", int(0.15 * nbPositions), 8),
    ("Scenario 2", int(0.15 * nbPositions), nbStudents),
    ("Scenario 3", nbPositions, 8),
    ("Scenario 4", nbPositions, nbStudents),
]


def truncate(prefA, prefB, sz):
    prA = [pr[:sz] for pr in prefA]
    prB = [
        [i for i in pr if j in prA[i]]  # todo: not efficient
        for j, pr in enumerate(prefB)
    ]
    return prA, prB


if __name__ == "__main__":
    logpop = pop.generate_logpop(nbPositions, nbStudents)
    prefP1, prefS1 = pop.draw_profile(logpop)

    with PdfPages("fig.pdf") as pdf:
        for iScenario, (name, sz1, sz2) in enumerate(scenarios):
            # truncate the preference of students
            prefS2, prefP2 = truncate(prefS1, prefP1, sz1)

            # truncate the preference of positions
            prefP3, prefS3 = truncate(prefP2, prefS2, sz2)

            # run deferred acceptance
            matchP, matchS = deferred_acceptance(prefP3, prefS3)

            #####################################

            cmap = mpl.cm.viridis
            norm = mpl.colors.Normalize(vmin=logpop.min(), vmax=logpop.max())
            txt1 = "First, students apply to %d positions" % sz1
            txt2 = "Then, positions keep at most %d applicants" % sz2

            #####################################

            plt.figure(figsize=(6, 5), tight_layout=True)
            plt.title(name)

            plt.imshow(logpop, origin="lower", norm=norm, cmap=cmap)
            plt.xlabel("students")
            plt.ylabel("positions")
            plt.colorbar()

            for s in range(nbStudents):
                if matchS[s] == None:
                    plt.plot([s], [nbPositions], "r.", markersize=3)
                else:
                    plt.plot([s], [matchS[s]], "r.", markersize=3)
            for p in range(nbPositions):
                if matchP[p] == None:
                    plt.plot([nbStudents], [p], "r.", markersize=3)

            plt.xlim((-0.5, nbStudents + 0.5))
            plt.ylim((-0.5, nbPositions + 0.5))

            pdf.savefig()
            plt.close()

            #####################################
            plt.figure(figsize=(10, 5), tight_layout=True)

            ax = plt.subplot(1, 2, 1)
            ax.title.set_text(txt1)

            tmp = logpop.copy()
            for p in range(nbPositions):
                l = set(range(nbStudents)) - set(prefP2[p])
                for s in l:
                    tmp[p, s] = None
                if prefP2[p] == []:
                    plt.axhline(p, color="r")

            plt.imshow(tmp, origin="lower", norm=norm, cmap=cmap)
            plt.xlabel("students")
            plt.ylabel("positions")

            ax = plt.subplot(1, 2, 2)
            ax.title.set_text(txt2)

            for s in range(nbStudents):
                l = set(range(nbPositions)) - set(prefS3[s])
                for p in l:
                    tmp[p, s] = None
                if prefS3[s] == []:
                    plt.axvline(s, color="r")

            plt.imshow(tmp, origin="lower", norm=norm, cmap=cmap)
            plt.xlabel("students")
            plt.ylabel("positions")

            pdf.savefig()
            plt.close()

            #####################################
            plt.figure(figsize=(10, 3), tight_layout=True)

            ax = plt.subplot(1, 2, 1)
            ax.title.set_text(txt1)

            YS = sorted([len(pr) for pr in prefS2], reverse=True)
            YP = sorted([len(pr) for pr in prefP2], reverse=True)
            plt.plot(YS, label="length of students' lists")
            plt.plot(YP, label="length of positions' lists")
            plt.ylim(bottom=0)
            plt.legend()

            ax = plt.subplot(1, 2, 2)
            ax.title.set_text(txt2)

            YS = sorted([len(pr) for pr in prefS3], reverse=True)
            YP = sorted([len(pr) for pr in prefP3], reverse=True)
            plt.plot(YS, label="length of students' lists")
            plt.plot(YP, label="length of positions' lists")
            plt.ylim(bottom=0)
            plt.legend()

            pdf.savefig()
            plt.close()
