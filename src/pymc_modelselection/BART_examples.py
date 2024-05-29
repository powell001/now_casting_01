from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as at
from numpy import random
#from myhelpers import printme
import xarray as xr
#from readinTransform import collecttransform
import pymc_bart as pmb

print(f"Running on PyMC v{pm.__version__}")

RANDOM_SEED = 5781
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")

# mytune = 1000
# mydraws = 5000
# myn_init = 1000
# mychains = 4
# mycores = 1

# dt1 = pd.read_csv(r"data\mergedDataforAnalysis_statespace_COMPLETE.csv", index_col=[0])

# print(dt1.head())
# print(dt1.columns)

# Y = dt1.iloc[:, 9].T.values
# X = dt1.iloc[:, [10,11,12]].values

# print(X.shape)
# print(Y.shape)

# if __name__ == "__main__":
#     with pm.Model() as bart_g:
#         mu = pmb.BART("mu", X, Y, m=50)
#         sigma = pm.HalfNormal("sigma", Y.std())
#         y = pm.Normal("y", mu, sigma, observed = Y)


#     with bart_g:
#         idata_bart_g = pm.sample (2000,tune=1000, cores=1)

# try:
#     coal = np.loadtxt(Path("..", "data", "coal.csv"))
# except FileNotFoundError:
#     coal = np.loadtxt(pm.get_data("coal.csv"))

# print(coal)

# # discretize data
# years = int(coal.max() - coal.min())
# bins = years // 4

# print(years, bins)

# hist, x_edges = np.histogram(coal, bins=bins)

# # compute the location of the centers of the discretized data
# x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2

# print(x_edges[:-1], x_edges[1], x_edges[0])

# # xdata needs to be 2D for BART
# x_data = x_centers[:, None]
# # express data as the rate number of disaster per year
# y_data = hist


# if __name__ == "__main__":

#     with pm.Model() as model_coal:
#         μ_ = pmb.BART("μ_", X=x_data, Y=np.log(y_data), m=20)
#         μ = pm.Deterministic("μ", pm.math.exp(μ_))
#         y_pred = pm.Poisson("y_pred", mu=μ, observed=y_data)
#         idata_coal = pm.sample(random_seed=RANDOM_SEED, cores=1)

#     _, ax = plt.subplots(figsize=(10, 6))

#     rates = idata_coal.posterior["μ"] / 4
#     rate_mean = rates.mean(dim=["draw", "chain"])
#     ax.plot(x_centers, rate_mean, "w", lw=3)
#     ax.plot(x_centers, y_data / 4, "k.")
#     az.plot_hdi(x_centers, rates, smooth=False)
#     az.plot_hdi(x_centers, rates, hdi_prob=0.5, smooth=False, plot_kwargs={"alpha": 0})
#     ax.plot(coal, np.zeros_like(coal) - 0.5, "k|")
#     ax.set_xlabel("years")
#     ax.set_ylabel("rate");

#     plt.show()

try:
    bikes = pd.read_csv(Path("..", "data", "bikes.csv"))
except FileNotFoundError:
    bikes = pd.read_csv(pm.get_data("bikes.csv"))

features = ["hour", "temperature", "humidity", "workingday"]

X = bikes[features]
Y = bikes["count"]

if __name__ == "__main__":
    with pm.Model() as model_bikes:
        α = pm.Exponential("α", 1)
        μ = pmb.BART("μ", X, np.log(Y), m=50)
        y = pm.NegativeBinomial("y", mu=pm.math.exp(μ), alpha=α, observed=Y)
        idata_bikes = pm.sample(compute_convergence_checks=False, random_seed=RANDOM_SEED, cores=1, chains=4)

    az.plot_trace(idata_bikes, var_names=["α"], kind="rank_bars");
    plt.show()

    pmb.plot_convergence(idata_bikes, var_name="μ");
    plt.show()

    pmb.plot_pdp(μ, X=X, Y=Y, grid=(2, 2), func=np.exp, var_discrete=[3]);
    plt.show()

    pmb.plot_variable_importance(idata_bikes, μ, X);
    plt.show()