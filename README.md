# simus_insp

This is one code to simulate INSP bipartite matching between students and open positions (mostky administartions).

## Content of the repository

The file `da.py` contains the Gale-Shapley (a.k.a. stable weddings) deferred acceptance algorithm.

The file `simulation.py` allows to launch a batch of 500 random experiments.
The details of the first experiment are output on stderr (screen).
Preferences of the candidates are inferred from the popularity model.
The strategic behaviour of students and employers is simulated.

The file `popularities.py` contains tools to infer the propularity model of the positions (e.g. Bercy is more popular than sous-prefecture de Haute-Savoie) from historical data about ENA competition.

The file `scenarios.py`can be used to run several scenarios and compare the outcomes graphically using
the output file `fig.pdf`.

## Running a simulations

To run a simulation in several scenarios `python3 simulation.py`

To change the parameters, edit the file.
