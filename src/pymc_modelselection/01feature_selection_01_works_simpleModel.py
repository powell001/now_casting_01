# prediction, out of sample: https://www.pymc-labs.com/blog-posts/out-of-model-predictions-with-pymc/

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as at
from numpy import random
from myhelpers import printme
import xarray as xr
from readinTransform import collecttransform

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)

az.style.use("arviz-plasmish")
# from cycler import cycler
# default_cycler = cycler(color=["#6a6a6a", "#bebebe", "#2a2eec",  "#000000"])
# plt.rc("axes", prop_cycle=default_cycler)
plt.rcParams['figure.figsize'] = [15, 7.5]
plt.rcParams['figure.dpi'] = 80

# import preliz as pz #
print(f"Running on PyMC v{pm.__version__}")

mytune = 1000
mydraws = 5000
myn_init = 1000
mychains = 2
mycores = 1

#########################################################################################
# collect from transformed data, see: "src\\pymc_modelselection\\readinTransform.py" 
#########################################################################################
df1 = collecttransform()

# NaNs
df1.dropna(inplace=True)
df1.to_csv("tmp22.csv")

printme(df1)

df1['gdp_total'].hist();
plt.title("Transformed NL Total GDP")
plt.savefig("fig\gdp_total")

df1 = df1.iloc[:, [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,47]] ############## Select Features

X = df1.copy()
y = X.pop("gdp_total")
N, D = X.shape

number1 = 115 ##################### number of observations, may not be N above; N may be less number of observations because N contains 'extended' data 
outsample = df1.shape[0] - number1

# Train ############
X_train = X.iloc[0:number1, :]
y_train = y.iloc[0:number1]

# Test ############
X_test = X.iloc[number1:, :]
y_test = y.iloc[number1:]

# Out sample ############
X_outsample = X.iloc[-outsample:, :]
y_outsample = y.iloc[-outsample:]

with  pm.Model(coords={"predictors": X.columns.values}) as random_model:
    # Prior on error SD
    sigma = pm.HalfNormal("sigma", 1)

    beta = pm.Normal("beta", 0, 1, dims="predictors")

     # No shrinkage on intercept
    alpha = pm.Normal("alpha", 0, 1)

    scores = pm.Normal("scores", alpha + at.dot(X.values, beta), sigma, observed=y.values)

    idata_random = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, cores=mycores, chains=mychains)
    idata_random.extend(pm.sample_posterior_predictive(idata_random))


### Model
graphvis = pm.model_to_graphviz(random_model)
graphvis.view()

####################
#### Prior data
####################
with random_model:
    prior_samples = pm.sample_prior_predictive(200)

az.plot_dist(
    df1['gdp_total'],
    kind="kde",
    color='red',
    hist_kwargs=dict(alpha=0.6),
    label="observed",
)
az.plot_dist(
    prior_samples.prior_predictive["scores"],
    kind="kde",
    color='blue',
    hist_kwargs=dict(alpha=0.6),
    label="prior predictive (simulated)",
)
plt.savefig("fig\plot_prior_predictives", dpi=75)
plt.show()
plt.close()

###################
##### Posteriors
###################
# These are posteriors
az.plot_trace(idata_random, combined=True)
plt.savefig("fig\plot_trace")
plt.show()
plt.close()

# Trace
# pm.summary(idata_random)
print(idata_random.keys())
print(idata_random.posterior)
print(idata_random.posterior['predictors'])

# Summary of posteriors
# az.summary(idata_random, round_to=5).to_csv("output_csvs_etc\posteriorSummary.csv")

# Plots of posteriors
# az.plot_posterior(idata_random)
# plt.savefig("fig\plot_posterior_random")
# plt.show()
# plt.close()

# Forest plots
az.plot_forest(idata_random, var_names=["beta"], rope=[-0.15, 0.15], combined=True, hdi_prob=0.95, r_hat=True);
plt.tight_layout()
plt.savefig("fig\plot_forest_allfeatures", dpi=75)

# # Bayesian posterior
az.plot_bpv(idata_random, var_names=["scores"], kind="p_value")
plt.savefig("fig\plot_bpv_pvalue", dpi=75)
plt.show()

az.plot_bpv(idata_random, var_names=["scores"], kind="t_stat")
plt.savefig("fig\plot_bpv_tstat", dpi=75)
plt.show()

az.plot_bpv(idata_random, var_names=["scores"], kind="u_value")
plt.savefig("fig\plot_bpv_uvalue", dpi=75)
plt.show()

# az.plot_ppc(idata_random, var_names=["scores"])
# plt.savefig("fig\plot_ppc", dpi=75)
# plt.show()

# Chains
az.plot_autocorr(idata_random, combined=True)
plt.show()

#############################################################
# Laplace
#############################################################

# with  pm.Model(coords={"predictors": X.columns.values}) as laplace_model:
#     # Prior on error SD
#     sigma = pm.HalfNormal("sigma", 1)

#     beta = pm.Normal("beta", 0, 1, dims="predictors")

#      # No shrinkage on intercept
#     alpha = pm.Normal("alpha", 0, 1)

#     mu = pm.Deterministic("mu", alpha + at.dot(X.values, beta))

#     scores = pm.Laplace("scores", mu=mu, b=1.0, observed=y.values)

#     idata_laplace = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, cores=mycores, chains=mychains)
#     idata_laplace.extend(pm.sample_posterior_predictive(idata_laplace))


##### Model
# graphvis = pm.model_to_graphviz(laplace_model)
# graphvis.view()

#####################
##### Prior data
#####################
# Prior
# with laplace_model:
#     prior_samples = pm.sample_prior_predictive(200)

# az.plot_dist(
#     df1['gdp_total'],
#     kind="kde",
#     color='red',
#     hist_kwargs=dict(alpha=0.6),
#     label="observed",
# )
# az.plot_dist(
#     prior_samples.prior_predictive["scores"],
#     kind="kde",
#     color='blue',
#     hist_kwargs=dict(alpha=0.6),
#     label="prior predictive (simulated)",
# )
# plt.savefig("fig\plot_prior_predictives", dpi=75)
# plt.show()
# plt.close()

# ##################
# #### Posteriors
# ##################

# az.plot_trace(idata_laplace, combined=True)
# plt.savefig("fig\plot_trace")
# plt.show()
# plt.close()

# # Trace
# pm.summary(idata_laplace)
# # print(idata_laplace.keys())
# # print(idata_laplace.posterior)
# # print(idata_laplace.posterior['predictors'])

# # Summary of posteriors
# az.summary(idata_laplace, round_to=5).to_csv("output_csvs_etc\posteriorSummary.csv")

# # # Plots of posteriors
# az.plot_posterior(idata_laplace)
# plt.savefig("fig\plot_posterior_random")
# plt.show()
# plt.close()

# # # Forest plots
# az.plot_forest(idata_laplace, var_names=["beta"], rope=[-0.15, 0.15], combined=True, hdi_prob=0.95, r_hat=True);
# plt.tight_layout()
# plt.savefig("fig\plot_forest", dpi=75)

# # # Bayesian posterior
# az.plot_bpv(idata_laplace, var_names=["scores"], kind="p_value")
# plt.savefig("fig\plot_bpv_pvalue", dpi=75)
# plt.show()

# az.plot_bpv(idata_laplace, var_names=["scores"], kind="t_stat")
# plt.savefig("fig\plot_bpv_tstat", dpi=75)
# plt.show()

# az.plot_bpv(idata_laplace, var_names=["scores"], kind="u_value")
# plt.savefig("fig\plot_bpv_uvalue", dpi=75)
# plt.show()

# az.plot_ppc(idata_laplace, var_names=["scores"])
# plt.savefig("fig\plot_ppc", dpi=75)
# plt.show()

# # Chains
# az.plot_autocorr(idata_laplace, combined=True)
# plt.show()


# ################################################################
# # Students T
# ################################################################
# with  pm.Model(coords={"predictors": X.columns.values}) as student_model:
#     # Prior on error SD
#     sigma = pm.HalfNormal("sigma", 1)

#     beta = pm.Normal("beta", 0, 1, dims="predictors")

#      # No shrinkage on intercept
#     alpha = pm.Normal("alpha", 0, 1)

#     mu = pm.Deterministic("mu", alpha + at.dot(X.values, beta))

#     scores = pm.StudentT("scores", mu=mu, sigma=sigma, nu=2, observed=y.values)

#     idata_student = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, cores=mycores, chains=mychains)
#     idata_student.extend(pm.sample_posterior_predictive(idata_student))

# #### Model
# graphvis = pm.model_to_graphviz(student_model)
# graphvis.view()

# #####################
# ##### Prior data
# #####################
# # Prior
# with student_model:
#     prior_samples = pm.sample_prior_predictive(200)

# az.plot_dist(
#     df1['gdp_total'],
#     kind="kde",
#     color='red',
#     hist_kwargs=dict(alpha=0.6),
#     label="observed",
# )
# az.plot_dist(
#     prior_samples.prior_predictive["scores"],
#     kind="kde",
#     color='blue',
#     hist_kwargs=dict(alpha=0.6),
#     label="prior predictive (simulated)",
# )
# plt.savefig("fig\plot_prior_predictives", dpi=75)
# plt.show()
# plt.close()

# ##################
# #### Posteriors
# ##################

# az.plot_trace(idata_student, combined=True)
# plt.savefig("fig\plot_trace")
# plt.show()
# plt.close()

# # Trace
# pm.summary(idata_student)
# # print(idata_laplace.keys())
# # print(idata_laplace.posterior)
# # print(idata_laplace.posterior['predictors'])

# # Summary of posteriors
# az.summary(idata_student, round_to=5).to_csv("output_csvs_etc\posteriorSummary.csv")

# # # Plots of posteriors
# az.plot_posterior(idata_student)
# plt.savefig("fig\plot_posterior_random")
# plt.show()
# plt.close()

# # # Forest plots
# az.plot_forest(idata_student, var_names=["beta"], rope=[-0.15, 0.15], combined=True, hdi_prob=0.95, r_hat=True);
# plt.tight_layout()
# plt.savefig("fig\plot_forest", dpi=75)

# # # Bayesian posterior
# az.plot_bpv(idata_student, var_names=["scores"], kind="p_value")
# plt.savefig("fig\plot_bpv_pvalue", dpi=75)
# plt.show()

# az.plot_bpv(idata_student, var_names=["scores"], kind="t_stat")
# plt.savefig("fig\plot_bpv_tstat", dpi=75)
# plt.show()

# az.plot_bpv(idata_student, var_names=["scores"], kind="u_value")
# plt.savefig("fig\plot_bpv_uvalue", dpi=75)
# plt.show()

# az.plot_ppc(idata_student, var_names=["scores"])
# plt.savefig("fig\plot_ppc", dpi=75)
# plt.show()

# # Chains
# az.plot_autocorr(idata_student, combined=True)
# plt.show()









# #################################################
# # ###############################################
# # # model comparison
# # ###############################################
# #################################################

# _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
# az.plot_ppc(idata_random, num_pp_samples=100, ax=axes[0], legend=False, colors=["C1", "C0", "C1"])
# axes[0].set_title("idata_random")
# az.plot_ppc(idata_laplace, num_pp_samples=100, ax=axes[1], colors=["C1", "C0", "C1"])
# axes[1].set_title(f"idata_laplace")
# az.plot_ppc(idata_laplace, num_pp_samples=100, ax=axes[1], colors=["C1", "C0", "C1"])
# axes[1].set_title(f"idata_student")
# plt.savefig("fig\comparisonone", dpi=75)
# plt.show()

# # ###############################################
# # # model comparison 2
# # ###############################################

# fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey="row")
# colors = ["C0", "C1", "C2"]
# titles = ["mean", "interquartil range"]
# modelos = ["idata_random", "idata_laplace", "idata_student"]
# idatas = [idata_random, idata_laplace, idata_student]

# def iqr(x, a=-1):
#     """interquartile range"""
#     return np.subtract(*np.percentile(x, [75, 25], axis=a))
# for idata, c in zip(idatas, colors):
#     az.plot_bpv(idata, kind="t_stat", t_stat="mean", ax=axes[0], color=c)
# for idata, c in zip(idatas, colors):
#     az.plot_bpv(idata, kind="t_stat", t_stat=iqr, ax=axes[1], color=c)

# for (
#     ax,
#     title,
# ) in zip(axes, titles):
#     ax.set_title(title)
#     for idx, (c, modelo) in enumerate(zip(colors, modelos)):
#         ax.legend_.legend_handles[idx]._alpha = 1
#         ax.legend_.legend_handles[idx]._color = c
#         ax.legend_._loc = 1
#         ax.legend_.texts[idx]._text = modelo + " " + ax.legend_.texts[idx]._text

# plt.savefig("fig\comparisontwo, dpi=75")

# fig, ax = plt.subplots(figsize=(10, 3))

# for idata, c in zip(idatas, colors):
#     az.plot_bpv(idata, color=c, ax=ax)

# plt.savefig("fig\comparisonfour", dpi=75)

# ###############################################
# # model comparison 3
# ###############################################

# waic_random = az.waic(idata_random)
# print(waic_random)

# waic_laplace = az.waic(idata_laplace)
# print(waic_laplace)

# waic_student = az.waic(idata_student)
# print(waic_student)

# loo_random = az.loo(idata_random)
# print(loo_random)

# loo_laplace = az.loo(idata_laplace)
# print(loo_laplace)

# loo_student = az.loo(idata_student)
# print(loo_student)

# cmp_df = az.compare({"idata_random": idata_random, "idata_laplace": idata_laplace, "idata_student": idata_student})
# # cmp_df.to_markdown()
# cmp_df.to_csv("output_csvs_etc\cmp_df.csv")
# print(cmp_df)


