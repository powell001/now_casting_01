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

#[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,47]
#df1 = df1.iloc[:, [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,46,47]] ############## Select Features
#df1 = df1.iloc[:, [0,1,2,6,7,11,12,18,28,40,41]] ############## Select Features
#df1 = df1.iloc[:, [0,1,2,4,9,13,47]] ############## Select Features

#featureLists = [0,1,2,4,6,7,9,11,12,13,14,15,17,18,19,20,22,24,26,27,28,29,31,32,33,37,38,39,40,41,47] ################################
featureLists = [0,1,2,4,6,7,9,11,12,15,18,22,24,26,28,29,37,38,39,40,41,47]
df1 = df1.iloc[:, featureLists]

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
with  pm.Model(coords={"predictors": X.columns.values}) as student_model1:
    # Prior on error SD
    sigma = pm.HalfNormal("sigma", 1)

    beta = pm.Normal("beta", 0, 1, dims="predictors")

    # No shrinkage on intercept
    alpha = pm.Normal("alpha", 0, 1)

    mu = pm.Deterministic("mu", alpha + at.dot(X.values, beta))

    scores = pm.StudentT("scores", mu=mu, sigma=sigma, nu=2, observed=y.values)

    idata_student1 = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, target_accept=0.98, cores=mycores, chains=mychains)
    idata_student1.extend(pm.sample_posterior_predictive(idata_student1))

#################################################################################
#################################################################################
#################################################################################

df1 = collecttransform()
df1.dropna(inplace=True)

#featureLists = [0,1,2,4,6,7,9,11,12,13,15,17,18,22,24,26,27,28,29,32,33,37,38,39,40,41,47]
featureLists = [0,1,2,4,6,7,9,11,12,15,18,22,24,26,28,29,37,38,39,40,41]

df1 = df1.iloc[:, featureLists]

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
with  pm.Model(coords={"predictors": X.columns.values}) as student_model2:
    # Prior on error SD
    sigma = pm.HalfNormal("sigma", 1)

    beta = pm.Normal("beta", 0, 1, dims="predictors")

    # No shrinkage on intercept
    alpha = pm.Normal("alpha", 0, 1)

    mu = pm.Deterministic("mu", alpha + at.dot(X.values, beta))

    scores = pm.StudentT("scores", mu=mu, sigma=sigma, nu=2, observed=y.values)

    idata_student2 = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, target_accept=0.98, cores=mycores, chains=mychains)
    idata_student2.extend(pm.sample_posterior_predictive(idata_student2))

# #################################################################################
# #################################################################################
# #################################################################################

# df1 = collecttransform()
# df1.dropna(inplace=True)

# featureLists = [0,1,2,4,6,7,9,11,12,15,18,22,24,26,28,29,37,38,39,40,41,47]
# df1 = df1.iloc[:, featureLists]

# X = df1.copy()
# y = X.pop("gdp_total")
# N, D = X.shape

# number1 = 115 ##################### number of observations, may not be N above; N may be less number of observations because N contains 'extended' data 
# outsample = df1.shape[0] - number1

# # Train ############
# X_train = X.iloc[0:number1, :]
# y_train = y.iloc[0:number1]

# # Test ############
# X_test = X.iloc[number1:, :]
# y_test = y.iloc[number1:]

# # Out sample ############
# X_outsample = X.iloc[-outsample:, :]
# y_outsample = y.iloc[-outsample:]

# ################################################################
# # Students T
# ################################################################
# with  pm.Model(coords={"predictors": X.columns.values}) as student_model3:
#     # Prior on error SD
#     sigma = pm.HalfNormal("sigma", 1)

#     beta = pm.Normal("beta", 0, 1, dims="predictors")

#     # No shrinkage on intercept
#     alpha = pm.Normal("alpha", 0, 1)

#     mu = pm.Deterministic("mu", alpha + at.dot(X.values, beta))

#     scores = pm.StudentT("scores", mu=mu, sigma=sigma, nu=2, observed=y.values)

#     idata_student3 = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, target_accept=0.98, cores=mycores, chains=mychains)
#     idata_student3.extend(pm.sample_posterior_predictive(idata_student3))

# # #################################################################################
# # #################################################################################
# # #################################################################################

# df1 = collecttransform()
# df1.dropna(inplace=True)

# featureLists = [0,1,2,4,6,7,9,11,12,15,18,22,24,26,28,29,37,38,39,40,41,47]
# df1 = df1.iloc[:, featureLists]

# X = df1.copy()
# y = X.pop("gdp_total")
# N, D = X.shape

# number1 = 115 ##################### number of observations, may not be N above; N may be less number of observations because N contains 'extended' data 
# outsample = df1.shape[0] - number1

# # Train ############
# X_train = X.iloc[0:number1, :]
# y_train = y.iloc[0:number1]

# # Test ############
# X_test = X.iloc[number1:, :]
# y_test = y.iloc[number1:]

# # Out sample ############
# X_outsample = X.iloc[-outsample:, :]
# y_outsample = y.iloc[-outsample:]

# ################################################################
# # Students T
# ################################################################
# with  pm.Model(coords={"predictors": X.columns.values}) as student_model4:
#     # Prior on error SD
#     sigma = pm.HalfNormal("sigma", 1)

#     beta = pm.Normal("beta", 0, 1, dims="predictors")

#     # No shrinkage on intercept
#     alpha = pm.Normal("alpha", 0, 1)

#     mu = pm.Deterministic("mu", alpha + at.dot(X.values, beta))

#     scores = pm.StudentT("scores", mu=mu, sigma=sigma, nu=2, observed=y.values)

#     idata_student4 = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, target_accept=0.98, cores=mycores, chains=mychains)
#     idata_student4.extend(pm.sample_posterior_predictive(idata_student4))


# # #################################################################################
# # #################################################################################
# # #################################################################################

# df1 = collecttransform()
# df1.dropna(inplace=True)

# featureLists = [0,1,2,6,7,11,12,24,26,38,39,40,41,47]

# df1 = df1.iloc[:, featureLists]

# X = df1.copy()
# y = X.pop("gdp_total")
# N, D = X.shape

# number1 = 115 ##################### number of observations, may not be N above; N may be less number of observations because N contains 'extended' data 
# outsample = df1.shape[0] - number1

# # Train ############
# X_train = X.iloc[0:number1, :]
# y_train = y.iloc[0:number1]

# # Test ############
# X_test = X.iloc[number1:, :]
# y_test = y.iloc[number1:]

# # Out sample ############
# X_outsample = X.iloc[-outsample:, :]
# y_outsample = y.iloc[-outsample:]

# ################################################################
# # Students T
# ################################################################
# with  pm.Model(coords={"predictors": X.columns.values}) as student_model5:
#     # Prior on error SD
#     sigma = pm.HalfNormal("sigma", 1)

#     beta = pm.Normal("beta", 0, 1, dims="predictors")

#     # No shrinkage on intercept
#     alpha = pm.Normal("alpha", 0, 1)

#     mu = pm.Deterministic("mu", alpha + at.dot(X.values, beta))

#     scores = pm.StudentT("scores", mu=mu, sigma=sigma, nu=2, observed=y.values)

#     idata_student5 = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, target_accept=0.98, cores=mycores, chains=mychains)
#     idata_student5.extend(pm.sample_posterior_predictive(idata_student5))

# # #################################################################################
# # #################################################################################
# # #################################################################################

# df1 = collecttransform()
# df1.dropna(inplace=True)

# featureLists = [0,1,2,6,7,11,12,39,40,47]
# df1 = df1.iloc[:, featureLists]

# X = df1.copy()
# y = X.pop("gdp_total")
# N, D = X.shape

# number1 = 115 ##################### number of observations, may not be N above; N may be less number of observations because N contains 'extended' data 
# outsample = df1.shape[0] - number1

# # Train ############
# X_train = X.iloc[0:number1, :]
# y_train = y.iloc[0:number1]

# # Test ############
# X_test = X.iloc[number1:, :]
# y_test = y.iloc[number1:]

# # Out sample ############
# X_outsample = X.iloc[-outsample:, :]
# y_outsample = y.iloc[-outsample:]

# ################################################################
# # Students T
# ################################################################
# with  pm.Model(coords={"predictors": X.columns.values}) as student_model6:
#     # Prior on error SD
#     sigma = pm.HalfNormal("sigma", 1)

#     beta = pm.Normal("beta", 0, 1, dims="predictors")

#     # No shrinkage on intercept
#     alpha = pm.Normal("alpha", 0, 1)

#     mu = pm.Deterministic("mu", alpha + at.dot(X.values, beta))

#     scores = pm.StudentT("scores", mu=mu, sigma=sigma, nu=2, observed=y.values)

#     idata_student6 = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, target_accept=0.98, cores=mycores, chains=mychains)
#     idata_student6.extend(pm.sample_posterior_predictive(idata_student6))


cmp_df = az.compare({"idata_student1": idata_student1, 
                     "idata_student2": idata_student2
                     })
# cmp_df.to_markdown()
cmp_df.to_csv("output_csvs_etc\cmp_df.csv")
print(cmp_df)

