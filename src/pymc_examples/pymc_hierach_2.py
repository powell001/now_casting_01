# https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/multilevel_modeling.html

import os
import warnings
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
import xarray as xr

warnings.filterwarnings("ignore", module="scipy")

print(f"Running on PyMC v{pm.__version__}")

RANDOM_SEED = 8924
az.style.use("arviz-darkgrid")

try:
    srrs2 = pd.read_csv(os.path.join("..", "data", "srrs2.dat"))
except FileNotFoundError:
    srrs2 = pd.read_csv(pm.get_data("srrs2.dat"))

srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2.state == "MN"].copy()

try:
    cty = pd.read_csv(os.path.join("..", "data", "cty.dat"))
except FileNotFoundError:
    cty = pd.read_csv(pm.get_data("cty.dat"))

srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
cty_mn = cty[cty.st == "MN"].copy()
cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips

srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
srrs_mn = srrs_mn.drop_duplicates(subset="idnum")
u = np.log(srrs_mn.Uppm).unique()

n = len(srrs_mn)

srrs_mn.county = srrs_mn.county.map(str.strip)
county, mn_counties = srrs_mn.county.factorize()
srrs_mn["county_code"] = county
radon = srrs_mn.activity
srrs_mn["log_radon"] = log_radon = np.log(radon + 0.1).values
floor_measure = srrs_mn.floor.values

# print(floor_measure)
# print(np.log(radon))

# srrs_mn.log_radon.hist(bins=25, grid=False)
# plt.xlabel("log(radon)")
# plt.ylabel("frequency");
# plt.show()

with pm.Model() as pooled_model:
    floor_ind = pm.MutableData("floor_ind", floor_measure, dims="obs_id")
        # where obs_id is the index? of each observation
        # Mutable: When making predictions or doing posterior predictive sampling, the shape of the registered data variable will most likely need to be changed.

    alpha = pm.Normal("alpha", 0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.Exponential("sigma", 5)
        # try changing sigma from 10 to 1 say

    theta = alpha + beta * floor_ind

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

# graphvis = pm.model_to_graphviz(pooled_model)
# graphvis.view()
# plt.show()

with pooled_model:
    prior_checks = pm.sample_prior_predictive(random_seed=RANDOM_SEED)

prior = prior_checks.prior.squeeze(drop=True)
print(prior)

# xr.concat((prior["alpha"], prior["alpha"] + prior["beta"]), dim="location").rename(
#     "log_radon"
# ).assign_coords(location=["basement", "floor"]).plot.scatter(
#     x="location", y="log_radon", edgecolors="none"
# );

# plt.show()
# plt.close()

with pooled_model:
    pooled_trace = pm.sample(random_seed=RANDOM_SEED, cores=1, chains=4)

#print(az.summary(pooled_trace, round_to=2))

post_mean = pooled_trace.posterior.mean(dim=("chain", "draw"))


# plt.scatter(srrs_mn.floor, np.log(srrs_mn.activity + 0.1))
# xvals = xr.DataArray(np.linspace(-0.2, 1.2))
# plt.plot(xvals, post_mean["beta"] * xvals + post_mean["alpha"], "r--");
# plt.show()
# plt.close()

###################
# not pooled
###################

coords = {"county": mn_counties}
#print(coords) # county names

with pm.Model(coords=coords) as unpooled_model:
    floor_ind = pm.MutableData("floor_ind", floor_measure, dims="obs_id")

    alpha = pm.Normal("alpha", 0, sigma=10, dims="county")
        # dimensions = county ensures that data is split by county
    beta = pm.Normal("beta", 0, sigma=10)
    sigma = pm.Exponential("sigma", 1)

    theta = alpha[county] + beta * floor_ind
        # alpha[county] intercept per county instead of combined

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

# graphvis = pm.model_to_graphviz(unpooled_model)
# graphvis.view()
# plt.show()

with unpooled_model:
    unpooled_trace = pm.sample(random_seed=RANDOM_SEED, cores=1, chains=4)


# ax = az.plot_forest(
#     unpooled_trace,
#     var_names=["alpha"],
#     r_hat=True,
#     combined=True,
#     figsize=(6, 18),
#     labeller=az.labels.NoVarLabeller(),
# )
# ax[0].set_ylabel("alpha");
# plt.show()
# plt.close()

# unpooled_means = unpooled_trace.posterior.mean(dim=("chain", "draw"))
# unpooled_hdi = az.hdi(unpooled_trace)

# unpooled_means_iter = unpooled_means.sortby("alpha")
# unpooled_hdi_iter = unpooled_hdi.sortby(unpooled_means_iter.alpha)

# _, ax = plt.subplots(figsize=(12, 5))
# xticks = np.arange(0, 86, 6)
# unpooled_means_iter.plot.scatter(x="county", y="alpha", ax=ax, alpha=0.8)
# ax.vlines(
#     np.arange(mn_counties.size),
#     unpooled_hdi_iter.alpha.sel(hdi="lower"),
#     unpooled_hdi_iter.alpha.sel(hdi="higher"),
#     color="orange",
#     alpha=0.6,
# )
# ax.set(ylabel="Radon estimate", ylim=(-2, 4.5))
# ax.set_xticks(xticks)
# ax.set_xticklabels(unpooled_means_iter.county.values[xticks])
# ax.tick_params(rotation=90);

# # plt.show()
# # plt.close()

# sample_counties = (
#     "LAC QUI PARLE",
#     "AITKIN",
#     "KOOCHICHING",
#     "DOUGLAS",
#     "CLAY",
#     "STEARNS",
#     "RAMSEY",
#     "ST LOUIS",
# )

# fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
# axes = axes.ravel()
# m = unpooled_means["beta"]
# for i, c in enumerate(sample_counties):
#     y = srrs_mn.log_radon[srrs_mn.county == c]
#     x = srrs_mn.floor[srrs_mn.county == c]
#     axes[i].scatter(x + np.random.randn(len(x)) * 0.01, y, alpha=0.4)

#     # No pooling model
#     b = unpooled_means["alpha"].sel(county=c)

#     # Plot both models and data
#     xvals = xr.DataArray(np.linspace(0, 1))
#     axes[i].plot(xvals, m * xvals + b)
#     axes[i].plot(xvals, post_mean["beta"] * xvals + post_mean["alpha"], "r--")
#     axes[i].set_xticks([0, 1])
#     axes[i].set_xticklabels(["basement", "floor"])
#     axes[i].set_ylim(-1, 3)
#     axes[i].set_title(c)
#     if not i % 2:
#         axes[i].set_ylabel("log radon level")

# # plt.show()
# # plt.close()

# with pm.Model(coords=coords) as partial_pooling:
#     county_idx = pm.MutableData("county_idx", county, dims="obs_id")

#     # Priors
#     mu_a = pm.Normal("mu_a", mu=0.0, sigma=10)
#     sigma_a = pm.Exponential("sigma_a", 1)

#     # Random intercepts
#     alpha = pm.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="county")

#     # Model error
#     sigma_y = pm.Exponential("sigma_y", 1)

#     # Expected value
#     y_hat = alpha[county_idx]

#     # Data likelihood
#     y_like = pm.Normal("y_like", mu=y_hat, sigma=sigma_y, observed=log_radon, dims="obs_id")

# # graphviz = pm.model_to_graphviz(partial_pooling)
# # graphviz.view()
# # print(graphviz)


# with partial_pooling:
#     partial_pooling_trace = pm.sample(tune=2000, random_seed=RANDOM_SEED, cores=1, chains=4)


# N_county = srrs_mn.groupby("county")["idnum"].count().values

# fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
# for ax, trace, level in zip(
#     axes,
#     (unpooled_trace, partial_pooling_trace),
#     ("no pooling", "partial pooling"),
# ):
#     # add variable with x values to xarray dataset
#     trace.posterior = trace.posterior.assign_coords({"N_county": ("county", N_county)})
#     # plot means
#     trace.posterior.mean(dim=("chain", "draw")).plot.scatter(
#         x="N_county", y="alpha", ax=ax, alpha=0.9
#     )
#     ax.hlines(
#         partial_pooling_trace.posterior.alpha.mean(),
#         0.9,
#         max(N_county) + 1,
#         alpha=0.4,
#         ls="--",
#         label="Est. population mean",
#     )

#     # plot hdi
#     hdi = az.hdi(trace).alpha
#     ax.vlines(N_county, hdi.sel(hdi="lower"), hdi.sel(hdi="higher"), color="orange", alpha=0.5)

#     ax.set(
#         title=f"{level.title()} Estimates",
#         xlabel="Nbr obs in county (log scale)",
#         xscale="log",
#         ylabel="Log radon",
#     )
#     ax.legend(fontsize=10)


# # plt.show()
# # plt.close()    

# with pm.Model(coords=coords) as varying_intercept:
#     floor_idx = pm.MutableData("floor_idx", floor_measure, dims="obs_id")
#     county_idx = pm.MutableData("county_idx", county, dims="obs_id")

#     # Priors
#     mu_a = pm.Normal("mu_a", mu=0.0, sigma=10.0)
#     sigma_a = pm.Exponential("sigma_a", 1)

#     # Random intercepts
#     alpha = pm.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="county")
#     # Common slope
#     beta = pm.Normal("beta", mu=0.0, sigma=10.0)

#     # Model error
#     sd_y = pm.Exponential("sd_y", 1)

#     # Expected value
#     y_hat = alpha[county_idx] + beta * floor_idx

#     # Data likelihood
#     y_like = pm.Normal("y_like", mu=y_hat, sigma=sd_y, observed=log_radon, dims="obs_id")

# graphviz = pm.model_to_graphviz(varying_intercept)
# graphviz.view()
# print(graphviz)

# with varying_intercept:
#     varying_intercept_trace = pm.sample(tune=2000, random_seed=RANDOM_SEED, cores=1, chains=4)

# ax = pm.plot_forest(
#     varying_intercept_trace,
#     var_names=["alpha"],
#     figsize=(6, 18),
#     combined=True,
#     r_hat=True,
#     labeller=az.labels.NoVarLabeller(),
# )
# ax[0].set_ylabel("alpha")

# # plt.show()
# # plt.close()   

# pm.plot_posterior(varying_intercept_trace, var_names=["sigma_a", "beta"]);
# # plt.show()
# # plt.close()

# print(az.summary(varying_intercept_trace, var_names=["beta"]))

# xvals = xr.DataArray([0, 1], dims="Level", coords={"Level": ["Basement", "Floor"]})
# post = varying_intercept_trace.posterior  # alias for readability
# theta = (
#     (post.alpha + post.beta * xvals).mean(dim=("chain", "draw")).to_dataset(name="Mean log radon")
# )

# _, ax = plt.subplots()
# theta.plot.scatter(x="Level", y="Mean log radon", alpha=0.2, color="k", ax=ax)  # scatter
# ax.plot(xvals, theta["Mean log radon"].T, "k-", alpha=0.2)
# # add lines too
# ax.set_title("MEAN LOG RADON BY COUNTY");

# plt.show()
# plt.close()

# sample_counties = (
#     "LAC QUI PARLE",
#     "AITKIN",
#     "KOOCHICHING",
#     "DOUGLAS",
#     "CLAY",
#     "STEARNS",
#     "RAMSEY",
#     "ST LOUIS",
# )

# fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
# axes = axes.ravel()
# m = unpooled_means["beta"]
# for i, c in enumerate(sample_counties):
#     y = srrs_mn.log_radon[srrs_mn.county == c]
#     x = srrs_mn.floor[srrs_mn.county == c]
#     axes[i].scatter(x + np.random.randn(len(x)) * 0.01, y, alpha=0.4)

#     # No pooling model
#     b = unpooled_means["alpha"].sel(county=c)

#     # Plot both models and data
#     xvals = xr.DataArray(np.linspace(0, 1))
#     axes[i].plot(xvals, m.values * xvals + b.values)
#     axes[i].plot(xvals, post_mean["beta"] * xvals + post_mean["alpha"], "r--")

#     varying_intercept_trace.posterior.sel(county=c).beta
#     post = varying_intercept_trace.posterior.sel(county=c).mean(dim=("chain", "draw"))
#     theta = post.alpha.values + post.beta.values * xvals
#     axes[i].plot(xvals, theta, "k:")
#     axes[i].set_xticks([0, 1])
#     axes[i].set_xticklabels(["basement", "floor"])
#     axes[i].set_ylim(-1, 3)
#     axes[i].set_title(c)
#     if not i % 2:
#         axes[i].set_ylabel("log radon level")

# plt.show()
# plt.close()


# with pm.Model(coords=coords) as varying_intercept_slope:
#     floor_idx = pm.MutableData("floor_idx", floor_measure, dims="obs_id")
#     county_idx = pm.MutableData("county_idx", county, dims="obs_id")

#     # Priors
#     mu_a = pm.Normal("mu_a", mu=0.0, sigma=10.0)
#     sigma_a = pm.Exponential("sigma_a", 1)

#     mu_b = pm.Normal("mu_b", mu=0.0, sigma=10.0)
#     sigma_b = pm.Exponential("sigma_b", 1)

#     # Random intercepts
#     alpha = pm.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="county")
#     # Random slopes
#     beta = pm.Normal("beta", mu=mu_b, sigma=sigma_b, dims="county")

#     # Model error
#     sigma_y = pm.Exponential("sigma_y", 1)

#     # Expected value
#     y_hat = alpha[county_idx] + beta[county_idx] * floor_idx

#     # Data likelihood
#     y_like = pm.Normal("y_like", mu=y_hat, sigma=sigma_y, observed=log_radon, dims="obs_id")


# graphviz = pm.model_to_graphviz(varying_intercept_slope)
# graphviz.show()

# with varying_intercept_slope:
#     varying_intercept_slope_trace = pm.sample(tune=2000, random_seed=RANDOM_SEED, cores=1, chains=4)


# fig, axs = plt.subplots(nrows=2)
# axs[0].plot(varying_intercept_slope_trace.posterior.sel(chain=0)["sigma_b"], alpha=0.5)
# axs[0].set(ylabel="sigma_b")
# axs[1].plot(varying_intercept_slope_trace.posterior.sel(chain=0)["beta"], alpha=0.05)
# axs[1].set(ylabel="beta");

# plt.show()
# plt.close()

# ax = az.plot_pair(
#     varying_intercept_slope_trace,
#     var_names=["beta", "sigma_b"],
#     coords=dict(county="AITKIN"),
#     marginals=True,
#     # marginal_kwargs={"kind": "hist"},
# )
# ax[1, 0].set_ylim(0, 0.7);

# plt.show()
# plt.close()