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

# collect from transformed data, see: "src\\pymc_modelselection\\readinTransform.py" 
df1, firstGDPlog = collecttransform()

# NaNs
df1.dropna(inplace=True)

printme(df1)

df1['gdp_total'].hist();
plt.title("Transformed NL Total GDP")
plt.savefig("fig\gdp_total")
plt.close()

#df1 = df1.iloc[:, [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]] ############## Select Features
#df1 = df1.iloc[:, [0,1,2,4,9,13,14,18,21,25,27,29,30,31,33,34,35,47,48,49,50,56]] ############## Select Features
df1 = df1.iloc[:, [0,6,7,11,12,20,24,28,40]] ############## Select Features

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

################################################################
# Normal Distribution
################################################################
with  pm.Model(coords={"predictors": X.columns.values}) as student_model:
    # Prior on error SD
    sigma = pm.HalfNormal("sigma", 1)

    beta = pm.Normal("beta", 0, 1, dims="predictors")

     # No shrinkage on intercept
    alpha = pm.Normal("alpha", 0, 1)

    mu = pm.Deterministic("mu", alpha + at.dot(X.values, beta))

    scores = pm.Normal("scores", mu=mu, sigma=sigma, observed=y.values)

    idata_student = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, cores=mycores, chains=mychains)
    idata_student.extend(pm.sample_posterior_predictive(idata_student))

    prior_samples = pm.sample_prior_predictive(200)


#### Model
# graphvis = pm.model_to_graphviz(random_model)
# graphvis.view()

####################
#### Prior data
####################
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

##################
#### Posteriors
##################
# These are posteriors
az.plot_trace(idata_student, var_names=['beta', 'alpha', 'sigma'], combined=True)
plt.savefig("fig\plot_trace")

# Trace
pm.summary(idata_student)
# print(idata_student.keys())
# print(idata_student.posterior)
# print(idata_student.posterior['predictors'])

# Summary of posteriors
az.summary(idata_student, round_to=5).to_csv("output_csvs_etc\posteriorSummary.csv")

# Plots of posteriors
az.plot_posterior(idata_student)
plt.savefig("fig\plot_posterior_random")

# Forest plots
az.plot_forest(idata_student, var_names=["beta"], rope=[-0.15, 0.15], combined=True, hdi_prob=0.95, r_hat=True);
plt.tight_layout()
plt.savefig("fig\plot_forest", dpi=75)

# Bayesian posterior
az.plot_bpv(idata_student, var_names=["scores"], kind="p_value")
plt.savefig("fig\plot_bpv_pvalue", dpi=75)

az.plot_bpv(idata_student, var_names=["scores"], kind="t_stat")
plt.savefig("fig\plot_bpv_tstat", dpi=75)

az.plot_bpv(idata_student, var_names=["scores"], kind="u_value")
plt.savefig("fig\plot_bpv_uvalue", dpi=75)

az.plot_ppc(idata_student, var_names=["scores"])
plt.savefig("fig\plot_ppc", dpi=75)

# Chains
az.plot_autocorr(idata_student, combined=True)
