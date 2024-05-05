# prediction, out of sample: https://www.pymc-labs.com/blog-posts/out-of-model-predictions-with-pymc/

########################################################
# Model Selection
########################################################

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

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 400)

az.style.use("arviz-plasmish")
# from cycler import cycler
# default_cycler = cycler(color=["#6a6a6a", "#bebebe", "#2a2eec",  "#000000"])
# plt.rc("axes", prop_cycle=default_cycler)
plt.rcParams['figure.figsize'] = [15, 7.5]
plt.rcParams['figure.dpi'] = 80

# import preliz as pz #
print(f"Running on PyMC v{pm.__version__}")

mytune = 500
mydraws = 2000
myn_init = None
mychains = 2
mycores = 1

df1, firstGDPlog = collecttransform()

df1['gdp_total'].hist();
plt.title("Transformed NL Total GDP")
plt.savefig("fig\gdp_total")

##############
# Feature selection using horseshoe prior
##############

df1 = df1.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,26,27,28,29,30]]
X = df1.copy()
y = X.pop("gdp_total")
N, D = X.shape

number1 = 109 ##################### number of observations, may not be N above; N may be less number of observations because N contains 'extended' data 
outsample = df1.shape[0] - number1

print(number1, outsample)

# Train ############
X_train = X.iloc[0:number1, :]
y_train = y.iloc[0:number1]

# Test ############
X_test = X.iloc[number1:, :]
y_test = y.iloc[number1:]

# Out sample ############
X_outsample = X.iloc[-outsample:, :]
y_outsample = y.iloc[-outsample:]

##### Horseshoe prior 
#see article for formulas
##global shrinkage
D0 = int(D/3) # changed to 3 from 2
#see article for formulas
##local shrinkage

with pm.Model(coords={"predictors": X.columns.values}) as test_score_model:

    # data containers for making predictions
    X = pm.MutableData("X", X_train.values)
    y = pm.MutableData("y", y_train.values)

    # Prior on error SD
    sigma = pm.HalfNormal("sigma", .5)

    # Global shrinkage prior
    tau = pm.HalfStudentT("tau", 2.0, D0/(D - D0)* sigma/ np.sqrt(N))
    # Local shrinkage prior
    lam = pm.HalfStudentT("lam", 5.0, dims="predictors")
    c2 = pm.InverseGamma("c2", 3, 1)
    z = pm.Normal("z", 0.0, 1.0, dims="predictors")
    # Shrunken coefficients
    beta = pm.Deterministic("beta", z * tau * lam * at.sqrt(c2 / (c2 + tau**2 * lam**2)), dims="predictors")
    # No shrinkage on intercept
    beta0 = pm.Normal("beta0", 0, 1.0)

    # Likelihood
    scores = pm.Normal("scores", beta0 + at.dot(X, beta), sigma, observed=y)

    # Sample posterior
    idata = pm.sample(draws = mydraws, chains=mychains, idata_kwargs={"log_likelihood": True}, target_accept=0.98, cores=mycores, return_inferencedata=True)

    idata.extend(pm.sample_prior_predictive(100))


# graphvis = pm.model_to_graphviz(test_score_model)
# graphvis.view()

# https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html

# dimensions:
# [chain, sample, shape variable]
# print(idata.posterior['beta0'].sel(draw=slice(0,4)))

# These are posteriors
az.plot_trace(idata, combined=True)
plt.show()

# Summary of posteriors
az.summary(idata, round_to=5).to_csv("posteriorSummary.csv")

# Prior
with test_score_model:
    prior_samples = pm.sample_prior_predictive(100)

# Prior, so before any data is incorporated
az.plot_dist(
    df1['gdp_total'],
    kind="hist",
    hist_kwargs=dict(alpha=0.6),
    label="observed",
)
plt.show()

az.plot_dist(
    prior_samples.prior_predictive["scores"],
    kind="kde",
    hist_kwargs=dict(alpha=0.6),
    label="simulated",
)
plt.show()


# az.plot_dist(
#     df1["gdp_total"].values,
#     kind="hist",
#     color="red",
#     hist_kwargs=dict(alpha=0.6),
#     label="observed",
# )
# az.plot_dist(
#     idata.prior_predictive["scores"],
#     kind="hist",
#     hist_kwargs=dict(alpha=0.6),
#     label="simulated",
# )
# plt.xticks(rotation=45);
# plt.tight_layout()
# plt.show()

# az.plot_trace(idata, var_names=["tau", "sigma", "c2"]);
# plt.tight_layout()
# plt.savefig("fig\plot_trace", dpi=75)

# az.plot_energy(idata);
# plt.savefig("fig\plot_energy")

# az.plot_forest(idata, var_names=["beta"], rope=[-0.005, 0.005], combined=True, hdi_prob=0.95, r_hat=True);
# plt.tight_layout()
# plt.savefig("fig\plot_forest", dpi=75)

# az.plot_posterior(idata, var_names=['z'])
# plt.savefig("fig\plot_posterior", dpi=75)

# # needs likelihood
# waic_l = az.waic(idata)
# print(waic_l)

# loo_l = az.loo(idata)
# print(loo_l)

# ########################################################
# # In sample on entire data set
# ########################################################

# with test_score_model:
#     pm.set_data({"X": X_train, "y": y_train}) ####################################### What do you want to forecast, insample of outsample
#     idata.extend(pm.sample_posterior_predictive(idata))

# # Compute the point prediction by taking the mean and defining the category via a threshold.
# p_test_pred = idata.posterior_predictive["scores"].mean(dim=["chain", "draw"])

# forecast1 = pd.DataFrame(p_test_pred.values.tolist(), columns=['forecast'])
# forecast1.to_csv("forecast.csv")

# dffed = forecast1.values.tolist()
# print(dffed)

# az.plot_posterior(idata.posterior_predictive["scores"])
# plt.savefig("fig\plot_posterior", dpi=75)

# # ### Final Plot
# def untransform(data, firstGDPlog, dffed):

#     # undiff then exp
#     gdp1 = df1['gdp_total'].dropna()
#     gdp2 = np.append(firstGDPlog, gdp1)
#     gdp3 = np.append(gdp2, dffed)
#     gdp4 = np.cumsum(gdp3)

#     x1 = gdp4.tolist()
#     y1 =np.arange(0, len(x1))
#     plt.close()
#     plt.plot(y1[:-2], x1[:-2], color='black')  # Plot the first part of the line in red 
#     plt.plot(y1[-3:], x1[-3:], color='red')  # Plot the second part of the line in blue 
#     plt.savefig("fig\final_forecast", dpi=75)

# untransform(df1, firstGDPlog, dffed)




# ########################################################
# # Simple Model
# ########################################################

# printme(df1)
# print(df1['gdp_total'].describe())

# selectGooduns = ['gdp_total','lag_gdp_total', 'MaandmutatieCPI_3_monthly', 'CHN_monthly', 'trend',
#                  'FinancieleSituatieKomende12Maanden_7_monthly','CPI_1_monthly',  'EconomischeSituatieKomende12Maanden_5_monthly']

# # selectGooduns = ['dummy_downturn', 'gdp_total', 'EconomischeSituatieLaatste12Maanden_4_monthly','lag_gdp_total', 'FinancieleSituatieLaatste12Maanden_6_monthly', 
# #                  'FinancieleSituatieKomende12Maanden_7_monthly','CPIAfgeleid_2_monthly']

# df2 = df1.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,26,27,28,29,30,36,37]]
# goodunscorr = df2.corr()
# goodunscorr.to_csv("output_csvs_etc\gooduns_corr_data.csv")
# df2.to_csv("output_csvs_etc\model_data.csv")

# ##############
# X = df2.copy()
# X["Intercept"] = np.ones(len(X))
# y = X.pop("gdp_total")
# N, D = X.shape
# print(N, D)

# number1 = 109

# # Train ############
# X_train = X.iloc[0:number1, :]
# y_train = y.iloc[0:number1]

# # Test ############
# X_test = X.iloc[number1:, :]
# y_test = y.iloc[number1:]

# print(X_train)
# print(y_train)

# # Forecast X data ############################

# ###################################################################################
# ###################################################################################

# assert N == X_train.shape[0] + X_test.shape[0]

# with pm.Model(coords={"predictors": X.columns.values}) as best_model1:
    
#     # data containers
#     X = pm.MutableData("X", X_train.values)
#     y = pm.MutableData("y", y_train.values)

#     # priors
#     betas = pm.Normal("betas", 0, 10, dims="predictors")
#     sigma = pm.HalfNormal("sigma", 10)
   
#     # linear model
#     mu = at.dot(X, betas)

#     # link function
#     # p = pm.Deterministic("p", mu)

#     # likelihood
#     outcome = pm.StudentT("obs", mu=mu, nu=4, sigma=sigma, observed=y)

#     # inference data
#     idata = pm.sample(draws = mydraws, n_init=myn_init, chains=mychains, cores=1, tune=mytune, return_inferencedata=True, idata_kwargs={'log_likelihood':True})

# with best_model1:
#     pm.set_data({"X": X_train, "y": y_train})
#     idata.extend(pm.sample_posterior_predictive(idata))

# # Compute the point prediction by taking the mean and defining the category via a threshold.
# p_test_pred = idata.posterior_predictive["obs"].mean(dim=["chain", "draw"])
# forecast1 = pd.DataFrame(p_test_pred.values.tolist(), columns=['forecast'])
# forecast1.to_csv("forecast.csv")

# forecast1.plot()
# df2['gdp_total'].plot()
# plt.show()

# # az.plot_posterior(idata.posterior_predictive["obs"])
# # plt.show()

# ########################################################
# # END: Simple Model
# ########################################################


# ########################################################
# # Compare Model
# ########################################################

# selectGooduns = ['dummy_downturn', 'gdp_total', 'EconomischeSituatieLaatste12Maanden_4_monthly','lag_gdp_total', 'FinancieleSituatieLaatste12Maanden_6_monthly', 'FinancieleSituatieKomende12Maanden_7_monthly','CPIAfgeleid_2_monthly', 'EA_monthly']
# df2 = df1.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,26,27,28,29,30,36,37]]

# df2.to_csv("output_csvs_etc\compare_model_data.csv")

# ##############
# X = df2.copy()
# X["Intercept"] = np.ones(len(X))
# y = X.pop("gdp_total")
# N, D = X.shape
# print(N, D)

# number1 = 109
# # Train ############
# X_train = X.iloc[0:number1, :]
# y_train = y.iloc[0:number1]

# # Test ############
# X_test = X.iloc[number1:, :]
# y_test = y.iloc[number1:]

# # Forecast X data ############################

# assert N == X_train.shape[0] + X_test.shape[0]

# with pm.Model(coords={"predictors": X.columns.values}) as Normal_model1:
    
#     # data containers
#     X = pm.MutableData("X", X_train.values)
#     y = pm.MutableData("y", y_train.values)

#     # priors
#     betas = pm.Normal("betas", 0, 10, dims="predictors")
#     sigma = pm.HalfNormal("sigma", 10)
   
#     # linear model
#     mu = at.dot(X, betas)

#     # link function
#     # p = pm.Deterministic("p", mu)

#     # likelihood
#     outcome = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

#     # inference data
#     idataC = pm.sample(draws = mydraws, n_init=myn_init, chains=mychains, cores=1, tune=mytune, return_inferencedata=True, idata_kwargs={'log_likelihood':True})

# print(az.summary(idataC, var_names=["betas"], round_to=2))
# az.plot_trace(idataC, var_names=["betas"], compact=True);
# plt.show()
# plt.close()

# print(az.summary(idataC))

# az.plot_posterior(
#     idataC, var_names=["betas"], figsize=(15, 4)
# );

# with Normal_model1:
#     pm.set_data({"X": X_train, "y": y_train})
#     idataC.extend(pm.sample_posterior_predictive(idataC))

# # Compute the point prediction by taking the mean and defining the category via a threshold.
# p_test_pred = idataC.posterior_predictive["obs"].mean(dim=["chain", "draw"])
# forecast1 = pd.DataFrame(p_test_pred.values.tolist(), columns=['forecast'])

# forecast1.to_csv("forecast_normal.csv")

# forecast1.plot()
# df2['gdp_total'].plot()
# plt.show()


# # az.plot_posterior(idataC.posterior_predictive["obs"])
# # plt.show()

# ###############################################
# # model comparison
# ###############################################

# _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
# az.plot_ppc(idata, num_pp_samples=100, ax=axes[0], legend=False, colors=["C1", "C0", "C1"])
# axes[0].set_title("StudentT model")
# az.plot_ppc(idataC, num_pp_samples=100, ax=axes[1], colors=["C1", "C0", "C1"])
# axes[1].set_title(f"Normal model")

# plt.savefig("fig\comparison1")

# ##################

# fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey="row")
# colors = ["C0", "C1"]
# titles = ["mean", "interquartil range"]
# modelos = ["StudentT model", f"Normal model"]
# idatas = [idata, idataC]


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

# plt.savefig("fig\lin-pol-bpv.png")

# ################

# fig, ax = plt.subplots(figsize=(10, 3))

# for idata, c in zip(idatas, colors):
#     az.plot_bpv(idata, color=c, ax=ax)

# plt.savefig("fig\lin-pol-bpv2.png")

# #################

# cmp_df = az.compare({"StudentT": idata, "Normal Model": idataC})
# # cmp_df.to_markdown()
# print(cmp_df)

# ##################

# az.plot_compare(cmp_df)
# plt.savefig("fig\compareplot.png")

###########################################################################
# Get Latest Data (we need the missing observations at the end of the dataset)
###########################################################################

#############################################################################################################################
# Run latest model
#############################################################################################################################


###########################################################################
###########################################################################
###########################################################################


# how much does adding Dummy change the forecast--removing the dummy changes gdp from negative to positive
# how much does adding trend change the forecast--removing dummy and trend (negative)
# same, but normal instead of studentt, trend as well (postitive)

# try a forecast
# selectGooduns = ['gdp_total', 'BeloningSeizoengecorrigeerd_2', 'Consumentenvertrouwen_1_monthly', 'trend',
#                  'EconomischeSituatieKomende12Maanden_5_monthly', 'FinancieleSituatieLaatste12Maanden_6_monthly','FinancieleSituatieKomende12Maanden_7_monthly',
#                    'CPI_1_monthly', 'ProducerConfidence_1_monthly', 'CHN_monthly',  'G20_monthly', 'EA_monthly', 'lag_gdp_total']
# #df2 = df1.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,26,27,28,29,30,36,37]]

# df2 = df1[selectGooduns]
# corr1 = df2.corr()
# corr1.to_csv('output_csvs_etc\correlations.csv')
# df2.to_csv("output_csvs_etc\premodel_data.csv")


# X = df2.copy()
# X["Intercept"] = np.ones(len(X))
# y = X.pop("gdp_total")
# N, D = X.shape
# print(N, D)

# number1 = 109
# # Train ############
# X_train = X.iloc[0:number1, :]
# y_train = y.iloc[0:number1]

# ### Forecast Data
# X_forecast = X.iloc[-2:,:]
# y_forecast = [0,0]

# print(X_forecast)


# with pm.Model(coords={"predictors": X.columns.values}) as Normal_model1:
    
#     # data containers
#     X = pm.MutableData("X", X_train.values)
#     y = pm.MutableData("y", y_train.values)

#     # priors
#     betas = pm.Normal("betas", 0, 1, dims="predictors")
#     sigma = pm.HalfNormal("sigma", 1)
   
#     # linear model
#     mu = at.dot(X, betas)

#     # link function
#     # p = pm.Deterministic("p", mu)

#     # nu distribution
#     #nu= pm.HalfNormal('nu', sigma=5)
    
#     # likelihood
#     # outcome = pm.StudentT("obs", mu=mu, nu=nu, sigma=sigma, observed=y)
#     outcome = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

#     # inference data
#     idataC = pm.sample(draws = mydraws, n_init=myn_init, chains=mychains, cores=1, tune=mytune, return_inferencedata=True, idata_kwargs={'log_likelihood':True})

# print(az.summary(idataC, var_names=["betas"], round_to=6))
# az.plot_trace(idataC, var_names=["betas"], compact=False);
# plt.show()
# plt.close()

# print(az.summary(idataC))

# az.plot_posterior(idataC, var_names=["betas"], figsize=(15, 4), hdi_prob=.80);

# with Normal_model1:
#     pm.set_data({"X": X_forecast, "y": y_forecast})
#     idataC.extend(pm.sample_posterior_predictive(idataC))

# # Compute the point prediction by taking the mean and defining the category via a threshold.
# p_test_pred = idataC.posterior_predictive["obs"].mean(dim=["chain", "draw"])
# forecast1 = pd.DataFrame(p_test_pred.values.tolist(), columns=['forecast'])
# print(forecast1)

# dffed = forecast1.values.tolist()
# print(dffed)

# az.plot_posterior(idataC.posterior_predictive["obs"])
# plt.show()


# ### Final Plot
# ### transform
# # undiff then exp
# gdp1 = df1['gdp_total'].dropna()
# gdp2 = np.append(firstGDPlog, gdp1)
# gdp3 = np.append(gdp2, dffed)
# gdp4 = np.cumsum(gdp3)

# print([np.exp(x) for x in gdp4])
# plt.close()
# x1 = gdp4.tolist()

# y1 =np.arange(0, len(x1))
# plt.plot(y1[:-2], x1[:-2], color='black')  # Plot the first part of the line in red 
# plt.plot(y1[-3:], x1[-3:], color='red')  # Plot the second part of the line in blue 
 
# plt.show() 