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

printme(df1)

df1['gdp_total'].hist();
plt.title("Transformed NL Total GDP")
plt.savefig("fig\gdp_total")


#df1 = df1.iloc[:, [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,46,47]] ############## Select Features
df1 = df1.iloc[:, [0,1,2,6,7,11,12,18,28,40,41]] ############## Select Features
#df1 = df1.iloc[:, [0,1,2,4,9,13,47]] ############## Select Features

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
# Students T
################################################################
with  pm.Model(coords={"predictors": X.columns.values}) as student_model:
    # Prior on error SD
    sigma = pm.HalfNormal("sigma", 1)

    beta = pm.Normal("beta", 0, 1, dims="predictors")

     # No shrinkage on intercept
    alpha = pm.Normal("alpha", 0, 1)

    mu = pm.Deterministic("mu", alpha + at.dot(X.values, beta))

    scores = pm.StudentT("scores", mu=mu, sigma=sigma, nu=2, observed=y.values)

    idata_student = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, target_accept=0.98, cores=mycores, chains=mychains)
    idata_student.extend(pm.sample_posterior_predictive(idata_student))

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

#########################################################################################
# collect from transformed data, see: "src\\pymc_modelselection\\readinTransform.py" 
#########################################################################################
df1 = collecttransform()

# NaNs
df1.dropna(inplace=True)

printme(df1)

df1['gdp_total'].hist();
plt.title("Transformed NL Total GDP")
plt.savefig("fig\gdp_total")

df1 = df1.iloc[:, [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,46,47]] ############## Select Features
#df1 = df1.iloc[:, [0,1,2,6,7,11,12,18,28,40,41]] ############## Select Features
#df1 = df1.iloc[:, [0,1,2,4,9,13,47]] ############## Select Features
#df1 = df1.iloc[:, [0,1,2,4,9,11,12,13,24,39,40,47,48]]

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
# Students T More Variables
################################################################
with  pm.Model(coords={"predictors": X.columns.values}) as student_model_more:
    # Prior on error SD
    sigma = pm.HalfNormal("sigma", 1)

    beta = pm.Normal("beta", 0, 1, dims="predictors")

     # No shrinkage on intercept
    alpha = pm.Normal("alpha", 0, 1)

    mu = pm.Deterministic("mu", alpha + at.dot(X.values, beta))

    scores = pm.StudentT("scores", mu=mu, sigma=sigma, nu=2, observed=y.values)

    idata_student_more = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, target_accept=0.98, cores=mycores, chains=mychains)
    idata_student_more.extend(pm.sample_posterior_predictive(idata_student_more))

################################################
###############################################
# model comparison
###############################################
################################################

_, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
az.plot_ppc(idata_student, num_pp_samples=100, ax=axes[0], legend=False, colors=["C1", "C0", "C1"])
axes[0].set_title("idata_student")
az.plot_ppc(idata_student_more, num_pp_samples=100, ax=axes[1], colors=["C1", "C0", "C1"])
axes[1].set_title(f"idata_student_more")
plt.savefig("fig\comparisonone", dpi=75)
plt.show()

###############################################
# model comparison 2
###############################################

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey="row")
colors = ["C0", "C1", "C2"]
titles = ["mean", "interquartil range"]
modelos = ["idata_student", "idata_student_more"]
idatas = [idata_student, idata_student_more]

def iqr(x, a=-1):
    """interquartile range"""
    return np.subtract(*np.percentile(x, [75, 25], axis=a))
for idata, c in zip(idatas, colors):
    az.plot_bpv(idata, kind="t_stat", t_stat="mean", ax=axes[0], color=c)
for idata, c in zip(idatas, colors):
    az.plot_bpv(idata, kind="t_stat", t_stat=iqr, ax=axes[1], color=c)

for (
    ax,
    title,
) in zip(axes, titles):
    ax.set_title(title)
    for idx, (c, modelo) in enumerate(zip(colors, modelos)):
        ax.legend_.legend_handles[idx]._alpha = 1
        ax.legend_.legend_handles[idx]._color = c
        ax.legend_._loc = 1
        ax.legend_.texts[idx]._text = modelo + " " + ax.legend_.texts[idx]._text

plt.savefig("fig\comparisontwo, dpi=75")

fig, ax = plt.subplots(figsize=(10, 3))

for idata, c in zip(idatas, colors):
    az.plot_bpv(idata, color=c, ax=ax)

plt.savefig("fig\comparisonfour", dpi=75)

###############################################
# model comparison 3
###############################################

waic_random = az.waic(idata_student)
print(waic_random)

waic_laplace = az.waic(idata_student)
print(waic_laplace)

loo_random = az.loo(idata_student_more)
print(loo_random)

loo_laplace = az.loo(idata_student_more)
print(loo_laplace)

cmp_df = az.compare({"idata_student": idata_student, "idata_student_more": idata_student_more})
# cmp_df.to_markdown()
cmp_df.to_csv("output_csvs_etc\cmp_df.csv")
print(cmp_df)

