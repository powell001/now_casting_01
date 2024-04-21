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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)

az.style.use("arviz-grayscale")
from cycler import cycler
default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
plt.rc("axes", prop_cycle=default_cycler)
plt.rc("figure", dpi=80)


# import preliz as pz #
print(f"Running on PyMC3 v{pm.__version__}")

mytune = 15000
mydraws = 18000
myn_init = 15000
mychains = 4

data1 = pd.read_csv("data\mergedDataforAnalysis.csv", index_col=[0])
gdp_total_original = data1['gdp_total']

numcols = data1.shape[1]

# add monthly (mo) to monthly data
month_columns = pd.read_csv("data\\a0_combinedMonthly.csv", index_col=[0])
data1.columns = [f'{i}_monthly' if i in month_columns else f'{i}' for i in data1.columns]

### Difference
nodiffthese = ['Bankruptcies_monthly', 'BusinessOutlook_Industry_monthly', 'BusinessOutlook_Retail_monthly', 'Consumentenvertrouwen_1_monthly',
               'EconomischKlimaat_2_monthly', 'Koopbereidheid_3_monthly', 'EconomischeSituatieLaatste12Maanden_4_monthly', 'EconomischeSituatieKomende12Maanden_5_monthly',
               'FinancieleSituatieLaatste12Maanden_6_monthly', 'FinancieleSituatieKomende12Maanden_7_monthly', 'GunstigeTijdVoorGroteAankopen_8_monthly', "CPI_1_monthly",
               'CPIAfgeleid_2_monthly', 'MaandmutatieCPI_3_monthly', 'MaandmutatieCPIAfgeleid_4_monthly', 'ProducerConfidence_1_monthly', 'ExpectedActivity_2_monthly', 
               'CHN_monthly', 'JPN_monthly', 'FRA_monthly', 'USA_monthly', 'DEU_monthly', 'CAN_monthly', 'G20_monthly', 'IMP_advanceEconomies_monthly', 'EXP_advancedEconomies_monthly', 
               'IMP_EuroArea_monthly', 'Exp_EuroArea_monthly', 'InterestRatesNLD_monthly', 'EA_monthly', 'US_monthly', 'UK_monthly', 'dummy_downturn']

diffthese = ['gdp_total', 'imports_goods_services', 'household_cons', 'gov_consumption', 'investments',
             'gpd_invest_business_households', 'gov_invest', 'change_supply', 'exports_goods_services',
             'BeloningSeizoengecorrigeerd_2', 'Loonkosten_7', 'BeloningVanWerknemers_8',
             'M3_1_monthly', 'M3_2_monthly', 'M1_monthly', 'AEX_close_monthly']

######################
# Dummy removed???
######################

assert numcols == len(nodiffthese) + len(diffthese) - 1 #dummy

# diff these
diff_data1 = data1.copy()
data1.to_csv("output_csvs_etc\datanodiff.csv")
data1[diffthese] = diff_data1[diffthese].diff()

# lag these
lagthese = ['imports_goods_services', 'household_cons', 'gov_consumption', 'investments',
             'gpd_invest_business_households', 'gov_invest', 'change_supply', 'exports_goods_services']

lag_data1 = data1.copy()
data1[lagthese] = lag_data1[lagthese].shift(1)
data1.columns = [f'lag_{i}' if i in lagthese else f'{i}' for i in data1.columns]

# lagged gdp_total but keep unlagged of course
data1['lag_gdp_total'] = data1['gdp_total'].shift(1)

printme(data1)

# correlations
corr1 = data1.corr()
corr1.to_csv('output_csvs_etc\correlations_all.csv')

##############################
# create 5 random features
##############################
rws = data1.shape[0]
x = pd.DataFrame(random.randint(100, size=(5, rws))).T
x.columns = ["random_" + str(x1)  for x1 in np.arange(0, 5)]
x.index = data1.index
data1 = data1.join(x)

### Normalize
normalized_data1 = (data1 - data1.mean())/data1.std()
#normalized_data1['gdp_total'] = data1['gdp_total']


### Diff Log gdp_total
normalized_data1['gdp_total'] = np.log(gdp_total_original).diff(1)

# ##############################
# # PYMC models
# ##############################
df1 = normalized_data1.copy()

selectthese = normalized_data1.columns # select all columns
df1 = df1[selectthese]
printme(df1)

##############
# add dummy
##############
# extreems dummy
df1['dummy_downturn'] = 0
df1.loc['2009-01-01', 'dummy_downturn'] = 1
df1.loc['2020-01-01', 'dummy_downturn'] = 1
df1.loc['2020-04-01', 'dummy_downturn'] = 1
# df1.loc['2020-07-01', 'dummy_downturn'] = 1
# df1.loc['2021-04-01', 'dummy_downturn'] = 1
# df1.loc['2021-07-01', 'dummy_downturn'] = 1

df1.to_csv("output_csvs_etc\df1.csv")

##############
# Examine model data
##############
too_few_obs = ['BusinessOutlook_Industry_monthly', 'BusinessOutlook_Retail_monthly', 'IMP_advanceEconomies_monthly', 'EXP_advancedEconomies_monthly', 'IMP_EuroArea_monthly', 'Exp_EuroArea_monthly', 'InterestRatesNLD_monthly']
df1.drop(columns = too_few_obs, inplace = True)
df1.dropna(inplace=True)
printme(df1)

corr1 = df1.corr()
corr1.to_csv('output_csvs_etc\correlations.csv')

df1.to_csv("output_csvs_etc\premodel_data.csv")
df1['gdp_total'].hist();
#plt.show()

# # #####
# # X = df1.copy()
# # y = X.pop("gdp_total")
# # N, D = X.shape

# # ##### Horseshoe prior 
# # #see article for formulas
# # ##global shrinkage
# # D0 = int(D/2)
# # #see article for formulas
# # ##local shrinkage


# # with pm.Model(coords={"predictors": X.columns.values}) as test_score_model:
# #     # Prior on error SD
# #     sigma = pm.HalfNormal("sigma", 50)

# #     # Global shrinkage prior
# #     tau = pm.HalfStudentT("tau", 2, D0/(D - D0)* sigma/ np.sqrt(N))
# #     # Local shrinkage prior
# #     lam = pm.HalfStudentT("lam", 5, dims="predictors")
# #     c2 = pm.InverseGamma("c2", 1, 1)
# #     z = pm.Normal("z", 0.0, 1.0, dims="predictors")
# #     # Shrunken coefficients
# #     beta = pm.Deterministic(
# #         "beta", z * tau * lam * at.sqrt(c2 / (c2 + tau**2 * lam**2)), dims="predictors"
# #     )
# #     # No shrinkage on intercept
# #     beta0 = pm.Normal("beta0", 100, 25.0)

# #     scores = pm.Normal("scores", beta0 + at.dot(X.values, beta), sigma, observed=y.values)

# # graphvis = pm.model_to_graphviz(test_score_model)
# # graphvis.view()

# # with test_score_model:
# #     prior_samples = pm.sample_prior_predictive(1000)

# # az.plot_dist(
# #     df1["gdp_total"].values,
# #     kind="hist",
# #     color="C1",
# #     hist_kwargs=dict(alpha=0.6),
# #     label="observed",
# # )
# # az.plot_dist(
# #     prior_samples.prior_predictive["scores"],
# #     kind="hist",
# #     hist_kwargs=dict(alpha=0.6),
# #     label="simulated",
# # )
# # plt.xticks(rotation=45);
# # plt.savefig("fig\plot_distribution")

# # with test_score_model:
# #     idata = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, target_accept=0.99, cores=1, chains=mychains)

# # ### model checking
# # az.plot_trace(idata, var_names=["tau", "sigma", "c2"]);
# # plt.savefig("fig\plot_trace")

# # az.plot_energy(idata);
# # plt.savefig("fig\plot_energy")

# # az.plot_forest(idata, var_names=["beta"], combined=True, hdi_prob=0.95, r_hat=True);
# # plt.savefig("fig\plot_forest")

# # az.plot_posterior(idata, var_names=['z'])
# # plt.savefig("fig\plot_posterior")

# # waic_l = az.waic(idata)
# # print(waic_l)

# # loo_l = az.loo(idata)
# # print(loo_l)

##############################################################
#####
selectGooduns = ['dummy_downturn', 'gdp_total', 'EconomischeSituatieLaatste12Maanden_4_monthly','lag_gdp_total', 'FinancieleSituatieLaatste12Maanden_6_monthly', 'FinancieleSituatieKomende12Maanden_7_monthly','CPIAfgeleid_2_monthly', 'EA_monthly']
df2 = df1[selectGooduns]

df2.to_csv("output_csvs_etc\model_data.csv")

##############
X = df2.copy()
X["Intercept"] = np.ones(len(X))
y = X.pop("gdp_total")
N, D = X.shape
print(N, D)

number1 = -2
# Train ############
X_train = X.iloc[0:number1, :]
y_train = y.iloc[0:number1]

# Test ############
X_test = X.iloc[number1:, :]
y_test = y.iloc[number1:]

# Forecast X data ############################

###################################################################################
###################################################################################

# assert N == X_train.shape[0] + X_test.shape[0]

# with pm.Model(coords={"predictors": X.columns.values}) as best_model1:
    
#     # data containers
#     X = pm.MutableData("X", X_train.values)
#     y = pm.MutableData("y", y_train.values)

#     # priors
#     betas = pm.Normal("betas", 0, 50, dims="predictors")
#     sigma = pm.HalfNormal("sigma", 50)
   
#     # linear model
#     mu = at.dot(X, betas)

#     # link function
#     # p = pm.Deterministic("p", mu)

#     # likelihood
#     outcome = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

#     # inference data
#     idata = pm.sample(draws = mydraws, n_init=myn_init, chains=mychains, cores=1, tune=mytune, return_inferencedata=True)

# print(az.summary(idata, var_names=["betas"], round_to=2))
# az.plot_trace(idata, var_names=["betas"], compact=True);
# plt.show()
# plt.close()

# print(az.summary(idata))

# az.plot_posterior(
#     idata, var_names=["betas"], figsize=(15, 4)
# );

# with best_model1:
#     pm.set_data({"X": X_test.values, "y": y_test})
#     idata.extend(pm.sample_posterior_predictive(idata))

# # Compute the point prediction by taking the mean and defining the category via a threshold.
# p_test_pred = idata.posterior_predictive["obs"].mean(dim=["chain", "draw"])
# print(p_test_pred)

# az.plot_posterior(idata.posterior_predictive["obs"])
# plt.show()

###################################################################################
###################################################################################



# with best_model1:
#     pm.sample_posterior_predictive(trace_1, extend_inferencedata=True)

# print(trace_1.posterior_predictive)

#print("posterior: ", trace_1.posterior)
# az.plot_ppc(trace_1, num_pp_samples=100);
# plt.show()
# plt.close()

# print(trace_1.obs)


#trace_1.posterior["y_model"] = trace_1.posterior["Intercept"] + trace_1.posterior["x"] * xr.DataArray(x)




# az.plot_forest(trace_1, var_names=["betas"], combined=True)
# plt.show()


# with best_model1:
#     pm.set_data({"X": X_train})
#     post_checks = pm.sample_posterior_predictive(trace_1)


# with model_1:
#     pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)
# idata = az.from_pymc3(trace=trace_1, posterior_predictive=post_checks)
# az.plot_ppc(idata, figsize=(12, 5));
# plt.show()


# with best_model1:
#     # update values of predictors:
#     pm.set_data({"X": X_test.values})
#     # use the updated values and predict outcomes and probabilities:
#     idata_2 = pm.sample_posterior_predictive(
#         idata_2,
#         var_names=["p"],
#         return_inferencedata=True,
#         predictions=True,
#         extend_inferencedata=True
#     )

# az.summary(trace_sf, round_to=2, var_names=["alpha", "betas"])

# print(idata_2.predictions)

# posterior_best = az.extract(idata_2)
# print(posterior_best)

# alpha_post = posterior_best['alpha'].mean().item()
# betas_post = posterior_best['betas'].mean().item()
# p_post = posterior_best['p'].mean().item()

# az.plot_trace(idata_2, var_names=["p"], compact=False);
# plt.show()

# print(alpha_post, betas_post, p_post)

#print(idata_2.predictions_constant_data.y) 

# print(idata_2.predictions_constant_data.sortby("pred")["pred"])

# graphvis = pm.model_to_graphviz(best_model1)
# graphvis.view()


    




# az.plot_trace(idata, var_names=["alpha","betas","sigma"], compact=False);
# plt.savefig("fig\plot_betas")

# print(az.summary(idata, var_names="betas"))

# az.plot_posterior(idata, var_names=["betas"], figsize=(15, 4));

# #########################
# #########################

# with best_model1:
#     pm.set_data({"X": X_test.values, "y": y_test})
#     idata.extend(pm.sample_posterior_predictive(idata))

# p_test_pred = idata.posterior_predictive["obs"].mean(dim=["chain", "draw"])
# y_test_pred = np.mean((p_test_pred).to_numpy())
# print("y_test_pred: ", y_test_pred)


# with best_model1:
#     # Configure sampler.
#     trace = pm.sample(5000, chains=mychains, cores=1, tune=1000, target_accept=0.98)
# az.plot_trace(data=trace);
# plt.show()

# az.summary(trace)
# plt.show()


# # this is the key, only X_test is specified, y_test (out of sample is calculated)


# with best_model1:
#     pm.set_data({"X": X_test}, model=best_model1)
#     ppc_test = pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# print(ppc_test.keys)
# print(p_test_pred = ppc_test["obs"].mean(axis=0))

#idata_best = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, cores=1, chains=mychains)

#az.summary(idata_best, var_names=["betas"], round_to=2)

# # az.plot_posterior(idata_best)
# # plt.savefig("fig\plot_posterior_best_one")
# az.plot_trace(idata_best);
# plt.show()

# print(idata_best.posterior_predictive)

# az.plot_ppc(idata_best, num_pp_samples=100);
# plt.show()

######################
# out of sample
######################
# RANDOM_SEED = 58
# rng = np.random.default_rng(RANDOM_SEED)

# predictors_out_of_sample = rng.normal(size=109)

# with best_model1:
#     # update values of predictors:
#     pm.set_data({"pred": predictors_out_of_sample})
#     # use the updated values and predict outcomes and probabilities:
#     idata_2 = pm.sample_posterior_predictive(
#         idata_best,
#         var_names=["p"],
#         return_inferencedata=True,
#         predictions=True,
#         extend_inferencedata=True,
#         random_seed=rng,
#     )


##################################
##################################

# df2 = df1.iloc[:, [0,3,5,7,11,13,15,17, -1]]
# X = df2.copy()
# y = X.pop("gdp_total")

# with  pm.Model(coords={"predictors": X.columns.values}) as random_model:
#     # Prior on error SD
#     sigma = pm.HalfNormal("sigma", 50)

#     beta = pm.Normal("beta", 0, 10, dims="predictors")

#      # No shrinkage on intercept
#     alpha = pm.Normal("alpha", 0, 10)

#     scores = pm.Normal("scores", alpha + at.dot(X.values, beta), sigma, observed=y.values)

#     idata_random = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, cores=1, chains=mychains)
#     idata_random.extend(pm.sample_posterior_predictive(idata_random))

# az.plot_posterior(idata_random)
# plt.savefig("plot_posterior_random")


# ###############################################
# # model comparison
# ###############################################

# _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
# az.plot_ppc(idata_best, num_pp_samples=100, ax=axes[0], legend=False, colors=["C1", "C0", "C1"])
# axes[0].set_title("best linear model")
# az.plot_ppc(idata_random, num_pp_samples=100, ax=axes[1], colors=["C1", "C0", "C1"])
# axes[1].set_title(f"random linear model")

# plt.savefig("comparison1")

# ##################

# fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey="row")
# colors = ["C0", "C1"]
# titles = ["mean", "interquartil range"]
# modelos = ["full model", f"small model"]
# idatas = [idata_best, idata_random]


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

# plt.savefig("lin-pol-bpv.png")

# ################

# fig, ax = plt.subplots(figsize=(10, 3))

# for idata, c in zip(idatas, colors):
#     az.plot_bpv(idata, color=c, ax=ax)

# plt.savefig("lin-pol-bpv2.png")


# #################

# cmp_df = az.compare({"model_best": idata_best, "model_random": idata_random})
# # cmp_df.to_markdown()
# print(cmp_df)


# ##################

# az.plot_compare(cmp_df)
# plt.savefig("compareplot.png")