# looking at similar! groups
# similarities, so don't use individual data, but not identical, so don't use group averages

# https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html#id19
# https://mc-stan.org/users/documentation/case-studies/pool-binary-trials.html
# https://mc-stan.org/users/documentation/case-studies/pool-binary-trials.html

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

data = pd.read_csv("src\pymc_examples\data\hierarch.txt", sep="\t")
at_bats, hits = data[["At-Bats", "Hits"]].to_numpy().T

N = len(hits)
player_names = data["FirstName"] + " " + data["LastName"]
coords = {"player_names": player_names.tolist()}

with pm.Model(coords=coords) as baseball_model:
    phi = pm.Uniform("phi", lower=0.0, upper=1.0)

    kappa_log = pm.Exponential("kappa_log", lam=1.5)
    kappa = pm.Deterministic("kappa", pt.exp(kappa_log))

    theta = pm.Beta("theta", alpha=phi * kappa, beta=(1.0 - phi) * kappa, dims="player_names")
    y = pm.Binomial("y", n=at_bats, p=theta, dims="player_names", observed=hits)

with baseball_model:
    theta_new = pm.Beta("theta_new", alpha=phi * kappa, beta=(1.0 - phi) * kappa)
    y_new = pm.Binomial("y_new", n=4, p=theta_new, observed=0)

graphvis = pm.model_to_graphviz(baseball_model)
graphvis.view()

with baseball_model:
    idata = pm.sample(2000, tune=2000, chains=2, target_accept=0.95, cores=1)

    # check convergence diagnostics
    assert all(az.rhat(idata) < 1.03)


az.plot_trace(idata, var_names=["phi", "kappa"]);
plt.show()

az.plot_forest(idata, var_names="theta");
plt.show()