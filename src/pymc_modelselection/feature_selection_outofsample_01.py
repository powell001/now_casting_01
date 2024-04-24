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

mytune = 5000
mydraws = 50000
myn_init = 1000
mychains = 4

# data1 = pd.read_csv("data\mergedDataforAnalysis.csv", index_col=[0])
# gdp_total_original = data1['gdp_total']

# numcols = data1.shape[1]

# # add monthly (mo) to monthly data
# month_columns = pd.read_csv("data\\a0_combinedMonthly.csv", index_col=[0])
# data1.columns = [f'{i}_monthly' if i in month_columns else f'{i}' for i in data1.columns]

# ### Difference
# nodiffthese = ['Bankruptcies_monthly', 'BusinessOutlook_Industry_monthly', 'BusinessOutlook_Retail_monthly', 'Consumentenvertrouwen_1_monthly',
#                'EconomischKlimaat_2_monthly', 'Koopbereidheid_3_monthly', 'EconomischeSituatieLaatste12Maanden_4_monthly', 'EconomischeSituatieKomende12Maanden_5_monthly',
#                'FinancieleSituatieLaatste12Maanden_6_monthly', 'FinancieleSituatieKomende12Maanden_7_monthly', 'GunstigeTijdVoorGroteAankopen_8_monthly', "CPI_1_monthly",
#                'CPIAfgeleid_2_monthly', 'MaandmutatieCPI_3_monthly', 'MaandmutatieCPIAfgeleid_4_monthly', 'ProducerConfidence_1_monthly', 'ExpectedActivity_2_monthly', 
#                'CHN_monthly', 'JPN_monthly', 'FRA_monthly', 'USA_monthly', 'DEU_monthly', 'CAN_monthly', 'G20_monthly', 'IMP_advanceEconomies_monthly', 'EXP_advancedEconomies_monthly', 
#                'IMP_EuroArea_monthly', 'Exp_EuroArea_monthly', 'InterestRatesNLD_monthly', 'EA_monthly', 'US_monthly', 'UK_monthly', 'dummy_downturn']

# diffthese = ['gdp_total', 'imports_goods_services', 'household_cons', 'gov_consumption', 'investments',
#              'gpd_invest_business_households', 'gov_invest', 'change_supply', 'exports_goods_services',
#              'BeloningSeizoengecorrigeerd_2', 'Loonkosten_7', 'BeloningVanWerknemers_8',
#              'M3_1_monthly', 'M3_2_monthly', 'M1_monthly', 'AEX_close_monthly']

# ######################
# # Dummy removed???
# ######################

# assert numcols == len(nodiffthese) + len(diffthese) - 1 #dummy

# # diff these
# diff_data1 = data1.copy()
# data1.to_csv("output_csvs_etc\datanodiff.csv")
# data1[diffthese] = diff_data1[diffthese].diff()

# # lag these (real values wont be available)
# lagthese = ['imports_goods_services', 'household_cons', 'gov_consumption', 'investments',
#              'gpd_invest_business_households', 'gov_invest', 'change_supply', 'exports_goods_services']

# lag_data1 = data1.copy()
# data1[lagthese] = lag_data1[lagthese].shift(1)
# data1.columns = [f'lag_{i}' if i in lagthese else f'{i}' for i in data1.columns]

# # lagged gdp_total but keep unlagged of course
# data1['lag_gdp_total'] = data1['gdp_total'].shift(1)

# printme(data1)

# # correlations
# corr1 = data1.corr()
# corr1.to_csv('output_csvs_etc\correlations_all.csv')

# ##############################
# # create 5 random features
# ##############################
# rws = data1.shape[0]
# x = pd.DataFrame(random.randint(100, size=(5, rws))).T
# x.columns = ["random_" + str(x1)  for x1 in np.arange(0, 5)]
# x.index = data1.index
# data1 = data1.join(x)

# ### Add trend ###
# data1['trend'] = np.arange(0, data1.shape[0])

# ### Normalize
# normalized_data1 = (data1 - data1.mean())/data1.std()
# #normalized_data1['gdp_total'] = data1['gdp_total']

# ### Diff Log gdp_total
# #normalized_data1['gdp_total'] = np.log(gdp_total_original)
# #normalized_data1['gdp_total'] = gdp_total_original.diff()
# normalized_data1['gdp_total'] = np.log(gdp_total_original).diff()

# # ##############################
# # # PYMC models
# # ##############################
# df1 = normalized_data1.copy()

# selectthese = normalized_data1.columns # select all columns
# df1 = df1[selectthese]
# printme(df1)

# ##############
# # add dummy
# ##############
# # extreems dummy
# df1['dummy_downturn'] = 0
# df1.loc['2009-01-01', 'dummy_downturn'] = 1
# df1.loc['2020-01-01', 'dummy_downturn'] = 1
# df1.loc['2020-04-01', 'dummy_downturn'] = 1
# # df1.loc['2020-07-01', 'dummy_downturn'] = 1
# # df1.loc['2021-04-01', 'dummy_downturn'] = 1
# # df1.loc['2021-07-01', 'dummy_downturn'] = 1

# df1.to_csv("output_csvs_etc\df1.csv")

# ##############
# # Examine model data
# ##############
# too_few_obs = ['BusinessOutlook_Industry_monthly', 'BusinessOutlook_Retail_monthly', 'IMP_advanceEconomies_monthly', 'EXP_advancedEconomies_monthly', 'IMP_EuroArea_monthly', 'Exp_EuroArea_monthly', 'InterestRatesNLD_monthly']
# df1.drop(columns = too_few_obs, inplace = True)
# df1.dropna(inplace=True)


# high_corr = ['MaandmutatieCPIAfgeleid_4_monthly', 'lag_imports_goods_services', 'EconomischKlimaat_2_monthly', 'Koopbereidheid_3_monthly', 'lag_gpd_invest_business_households',
#              'EconomischeSituatieLaatste12Maanden_4_monthly', 'GunstigeTijdVoorGroteAankopen_8_monthly', 'M3_2_monthly', 'UK_monthly', 'CPIAfgeleid_2_monthly', 'ExpectedActivity_2_monthly']

# df1.drop(columns = high_corr, inplace = True)
# printme(df1)

# corr1 = df1.corr()
# corr1.to_csv('output_csvs_etc\correlations.csv')

# df1.to_csv("output_csvs_etc\premodel_data.csv")

# df1['gdp_total'].hist();
# plt.title("Transformed NL Total GDP")
# plt.savefig("fig\gdp_total")

# # ##############
# # # Feature selection
# # ##############

# # df1 = df1.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,26,27,28,29,30]]
# # X = df1.copy()
# # y = X.pop("gdp_total")
# # N, D = X.shape

# # number1 = 109
# # # Train ############
# # X_train = X.iloc[0:number1, :]
# # y_train = y.iloc[0:number1]

# # # Test ############
# # X_test = X.iloc[number1:, :]
# # y_test = y.iloc[number1:]

# # ##### Horseshoe prior 
# # #see article for formulas
# # ##global shrinkage
# # D0 = int(D/2)
# # #see article for formulas
# # ##local shrinkage

# # with pm.Model(coords={"predictors": X.columns.values}) as test_score_model:

# #     # data containers for making predictions
# #     X = pm.MutableData("X", X_train.values)
# #     y = pm.MutableData("y", y_train.values)

# #     # Prior on error SD
# #     sigma = pm.HalfNormal("sigma", 25.0)

# #     # Global shrinkage prior
# #     tau = pm.HalfStudentT("tau", 30.0, D0/(D - D0)* sigma/ np.sqrt(N))
# #     # Local shrinkage prior
# #     lam = pm.HalfStudentT("lam", 30.0, dims="predictors")
# #     c2 = pm.InverseGamma("c2", 3.0, 0.5)
# #     z = pm.Normal("z", 0.0, 30.0, dims="predictors")
# #     # Shrunken coefficients
# #     beta = pm.Deterministic(
# #         "beta", z * tau * lam * at.sqrt(c2 / (c2 + tau**2 * lam**2)), dims="predictors"
# #     )
# #     # No shrinkage on intercept
# #     beta0 = pm.Normal("beta0", 10, 25.0)

# #     # Likelihood
# #     scores = pm.Normal("scores", beta0 + at.dot(X, beta), sigma, observed=y)

# # # graphvis = pm.model_to_graphviz(test_score_model)
# # # graphvis.view()

# # with test_score_model:
# #     prior_samples = pm.sample_prior_predictive()

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
# #     idata = pm.sample(draws = mydraws, n_init=myn_init, chains=mychains, idata_kwargs={"log_likelihood": True}, target_accept=0.98, cores=1, return_inferencedata=True)

# # ### model checking
# # az.plot_trace(idata, var_names=["tau", "sigma", "c2"]);
# # plt.tight_layout()
# # plt.savefig("fig\plot_trace", dpi=75)

# # az.plot_energy(idata);
# # plt.savefig("fig\plot_energy")

# # az.plot_forest(idata, var_names=["beta"], rope=[-0.005, 0.005], combined=True, hdi_prob=0.95, r_hat=True);
# # plt.tight_layout()
# # plt.savefig("fig\plot_forest", dpi=75)

# # az.plot_posterior(idata, var_names=['z'])
# # plt.savefig("fig\plot_posterior", dpi=75)

# # # needs likelihood
# # waic_l = az.waic(idata)
# # print(waic_l)

# # loo_l = az.loo(idata)
# # print(loo_l)

# # ########################################################
# # # In sample on entire data set
# # ########################################################

# # with test_score_model:
# #     pm.set_data({"X": X_train, "y": y_train})
# #     idata.extend(pm.sample_posterior_predictive(idata))

# # # Compute the point prediction by taking the mean and defining the category via a threshold.
# # p_test_pred = idata.posterior_predictive["scores"].mean(dim=["chain", "draw"])

# # forecast1 = pd.DataFrame(p_test_pred.values.tolist(), columns=['forecast'])
# # forecast1.to_csv("forecast.csv")

# # forecast1.plot()
# # df1['gdp_total'].plot()
# # plt.show()


# # az.plot_posterior(idata.posterior_predictive["scores"])
# # plt.savefig("fig\plot_posterior", dpi=75)


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

##############
# SAME AS ABOVE
##############

# use C:\Users\jpark\VSCode\now_casting_01\src\state_space_python\local_linear_trend_data_quarterly
# to extend gdp data

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

# lag these (real values wont be available)
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

### Add trend ###
data1['trend'] = np.arange(0, data1.shape[0])

### Normalize
normalized_data1 = (data1 - data1.mean())/data1.std()
#normalized_data1['gdp_total'] = data1['gdp_total']

### Diff Log gdp_total
#normalized_data1['gdp_total'] = np.log(gdp_total_original)
#normalized_data1['gdp_total'] = gdp_total_original.diff()
firstGDPlog = np.log(gdp_total_original).values[0] # needed to reconstruct
normalized_data1['gdp_total'] = np.log(gdp_total_original).diff()

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


high_corr = ['MaandmutatieCPIAfgeleid_4_monthly', 'lag_imports_goods_services', 'EconomischKlimaat_2_monthly', 'Koopbereidheid_3_monthly', 'lag_gpd_invest_business_households',
             'EconomischeSituatieLaatste12Maanden_4_monthly', 'GunstigeTijdVoorGroteAankopen_8_monthly', 'M3_2_monthly', 'UK_monthly', 'CPIAfgeleid_2_monthly', 'ExpectedActivity_2_monthly']

df1.drop(columns = high_corr, inplace = True)
printme(df1)

corr1 = df1.corr()
corr1.to_csv('output_csvs_etc\correlations.csv')

df1.to_csv("output_csvs_etc\premodel_data.csv")

###########################################################################
###########################################################################
###########################################################################


# how much does adding Dummy change the forecast--removing the dummy changes gdp from negative to positive
# how much does adding trend change the forecast--removing dummy and trend (negative)
# same, but normal instead of studentt, time as well

# try a forecast
selectGooduns = ['gdp_total', 'BeloningSeizoengecorrigeerd_2', 'Consumentenvertrouwen_1_monthly', 'trend',
                 'EconomischeSituatieKomende12Maanden_5_monthly', 'FinancieleSituatieLaatste12Maanden_6_monthly','FinancieleSituatieKomende12Maanden_7_monthly',
                   'CPI_1_monthly', 'ProducerConfidence_1_monthly', 'CHN_monthly',  'G20_monthly', 'EA_monthly', 'lag_gdp_total']
#df2 = df1.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,26,27,28,29,30,36,37]]

df2 = df1[selectGooduns]
corr1 = df2.corr()
corr1.to_csv('output_csvs_etc\correlations.csv')
df2.to_csv("output_csvs_etc\premodel_data.csv")


X = df2.copy()
X["Intercept"] = np.ones(len(X))
y = X.pop("gdp_total")
N, D = X.shape
print(N, D)

number1 = 109
# Train ############
X_train = X.iloc[0:number1, :]
y_train = y.iloc[0:number1]

### Forecast Data
X_forecast = X.iloc[-2:,:]
y_forecast = [0,0]

print(X_forecast)


with pm.Model(coords={"predictors": X.columns.values}) as Normal_model1:
    
    # data containers
    X = pm.MutableData("X", X_train.values)
    y = pm.MutableData("y", y_train.values)

    # priors
    betas = pm.Normal("betas", 0, 1, dims="predictors")
    sigma = pm.HalfNormal("sigma", 1)
   
    # linear model
    mu = at.dot(X, betas)

    # link function
    # p = pm.Deterministic("p", mu)

    # nu distribution
    #nu= pm.HalfNormal('nu', sigma=5)
    
    # likelihood
    # outcome = pm.StudentT("obs", mu=mu, nu=nu, sigma=sigma, observed=y)
    outcome = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

    # inference data
    idataC = pm.sample(draws = mydraws, n_init=myn_init, chains=mychains, cores=1, tune=mytune, return_inferencedata=True, idata_kwargs={'log_likelihood':True})

print(az.summary(idataC, var_names=["betas"], round_to=6))
az.plot_trace(idataC, var_names=["betas"], compact=False);
plt.show()
plt.close()

print(az.summary(idataC))

az.plot_posterior(idataC, var_names=["betas"], figsize=(15, 4), hdi_prob=.80);

with Normal_model1:
    pm.set_data({"X": X_forecast, "y": y_forecast})
    idataC.extend(pm.sample_posterior_predictive(idataC))

# Compute the point prediction by taking the mean and defining the category via a threshold.
p_test_pred = idataC.posterior_predictive["obs"].mean(dim=["chain", "draw"])
forecast1 = pd.DataFrame(p_test_pred.values.tolist(), columns=['forecast'])
print(forecast1)

dffed = forecast1.values.tolist()
print(dffed)

az.plot_posterior(idataC.posterior_predictive["obs"])
plt.show()

### transform
# undiff then exp
gdp1 = df1['gdp_total'].dropna()
gdp2 = np.append(firstGDPlog, gdp1)
gdp3 = np.append(gdp2, dffed)
gdp4 = np.cumsum(gdp3)

print([np.exp(x) for x in gdp4])
plt.close()
x1 = gdp4.tolist()

y1 =np.arange(0, len(x1))
plt.plot(y1[:-2], x1[:-2], color='black')  # Plot the first part of the line in red 
plt.plot(y1[-3:], x1[-3:], color='red')  # Plot the second part of the line in blue 
 
plt.show() 