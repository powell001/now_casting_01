import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from numpy import random
from myhelpers import printme

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

az.style.use("arviz-grayscale")
from cycler import cycler
default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
plt.rc("axes", prop_cycle=default_cycler)
plt.rc("figure", dpi=300)


# import preliz as pz #
print(f"Running on PyMC3 v{pm.__version__}")

mytune = 1000
mydraws = 500
myn_init = 5000

data1 = pd.read_csv("data\mergedDataforAnalysis.csv", index_col=[0])
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

assert numcols == len(nodiffthese) + len(diffthese) - 1 #dummy

# diff these
diff_data1 = data1.copy()
data1.to_csv("datanodiff.csv")
data1[diffthese] = diff_data1[diffthese].diff()

# lag these
lagthese = ['imports_goods_services', 'household_cons', 'gov_consumption', 'investments',
             'gpd_invest_business_households', 'gov_invest', 'change_supply', 'exports_goods_services']

lag_data1 = data1.copy()
data1[lagthese] = lag_data1[lagthese].shift(1)
data1.columns = [f'lag_{i}' if i in lagthese else f'{i}' for i in data1.columns]
printme(data1)

# correlations
corr1 = data1.corr()
corr1.to_csv('correlations_all.csv')

##############################
# create 5 random features
##############################
# rws = data1.shape[0]
# x = pd.DataFrame(random.randint(100, size=(5, rws))).T
# x.columns = ["random_" + str(x1)  for x1 in np.arange(0, 5)]
# x.index = data1.index
# data1 = data1.join(x)

### Normalize
normalized_data1 = (data1 - data1.mean())/data1.std()
printme(normalized_data1)

### Log gdp_total
#normalized_data1['gdp_total'] = np.log(normalized_data1['gdp_total'])

# ##############################
# # PYMC models
# ##############################
df1 = normalized_data1.copy()

selectthese = normalized_data1.columns
df1 = df1[selectthese]

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

df1.to_csv("df1.csv")

##############
# Examine model data
##############
too_few_obs = ['BusinessOutlook_Industry_monthly', 'BusinessOutlook_Retail_monthly', 'IMP_advanceEconomies_monthly', 'EXP_advancedEconomies_monthly', 'IMP_EuroArea_monthly', 'Exp_EuroArea_monthly', 'InterestRatesNLD_monthly']
df1.drop(columns = too_few_obs, inplace = True)
df1.dropna(inplace=True)
printme(df1)

corr1 = df1.corr()
corr1.to_csv('correlations.csv')

df1.to_csv("premodel_data.csv")
df1['gdp_total'].hist();
#plt.show()

#####
X = df1.copy()
y = X.pop("gdp_total")
N, D = X.shape

##### Horseshoe prior 
#see article for formulas
##global shrinkage
D0 = int(D/2)
#see article for formulas
##local shrinkage
import pytensor.tensor as at

with pm.Model(coords={"predictors": X.columns.values}) as test_score_model:
    # Prior on error SD
    sigma = pm.HalfNormal("sigma", 50)

    # Global shrinkage prior
    tau = pm.HalfStudentT("tau", 2, D0/(D - D0)* sigma/ np.sqrt(N))
    # Local shrinkage prior
    lam = pm.HalfStudentT("lam", 5, dims="predictors")
    c2 = pm.InverseGamma("c2", 1, 1)
    z = pm.Normal("z", 0.0, 1.0, dims="predictors")
    # Shrunken coefficients
    beta = pm.Deterministic(
        "beta", z * tau * lam * at.sqrt(c2 / (c2 + tau**2 * lam**2)), dims="predictors"
    )
    # No shrinkage on intercept
    beta0 = pm.Normal("beta0", 100, 25.0)

    scores = pm.Normal("scores", beta0 + at.dot(X.values, beta), sigma, observed=y.values)

graphvis = pm.model_to_graphviz(test_score_model)
graphvis.view()

with test_score_model:
    prior_samples = pm.sample_prior_predictive(1000)

az.plot_dist(
    df1["gdp_total"].values,
    kind="hist",
    color="C1",
    hist_kwargs=dict(alpha=0.6),
    label="observed",
)
az.plot_dist(
    prior_samples.prior_predictive["scores"],
    kind="hist",
    hist_kwargs=dict(alpha=0.6),
    label="simulated",
)
plt.xticks(rotation=45);
plt.savefig("plot_distribution")

with test_score_model:
    idata = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, target_accept=0.99, cores=1, chains=4)

### model checking
az.plot_trace(idata, var_names=["tau", "sigma", "c2"]);
plt.savefig("plot_trace")

az.plot_energy(idata);
plt.savefig("plot_energy")

az.plot_forest(idata, var_names=["beta"], combined=True, hdi_prob=0.95, r_hat=True);
plt.savefig("plot_forest")

az.plot_posterior(idata, var_names=['z'])
plt.savefig("plot_posterior")

waic_l = az.waic(idata)
print(waic_l)

loo_l = az.loo(idata)
print(loo_l)

# with pm.Model() as model_one:
#     alpha = pm.Normal("alpha", mu=0, sigma=10)
    
#     # monthly
#     household_cons_name = pm.Normal("household_cons", mu=0, sigma=10)
#     # Consumentenvertrouwen_1_name = pm.Normal("Consumentenvertrouwen_1", mu=0, sigma=10)
#     # Koopbereidheid_3_name = pm.Normal("Koopbereidheid_3", mu=0, sigma=10)
#     # EconomischeSituatieKomende12Maanden_5_name = pm.Normal("EconomischeSituatieKomende12Maanden_5", mu=0, sigma=10)
#     # M3_2_name = pm.Normal("M3_2", mu=0, sigma=10)

#     # Koopbereidheid_3_name = pm.Normal("Koopbereidheid_3", mu=0, sigma=10)
#     # EconomischeSituatieLaatste12Maanden_4_name = pm.Normal("EconomischeSituatieLaatste12Maanden_4", mu=0, sigma=10)
    
#     # FinancieleSituatieLaatste12Maanden_6_name = pm.Normal("FinancieleSituatieLaatste12Maanden_6", mu=0, sigma=10)
#     # DEU_name = pm.Normal("DEU", mu=0, sigma=10)
#     # CPI_1_name = pm.Normal("CPI_1", mu=0, sigma=10)
#     # GunstigeTijdVoorGroteAankopen_8_name = pm.Normal("GunstigeTijdVoorGroteAankopen_8", mu=0, sigma=10)
#     # ProducerConfidence_1_name = pm.Normal("ProducerConfidence_1", mu=0, sigma=10)
    
#     # # quarter
#     # BeloningSeizoengecorrigeerd_2_name = pm.Normal("BeloningSeizoengecorrigeerd_2", mu=0, sigma=10)
#     # Loonkosten_7_name = pm.Normal("Loonkosten_7", mu=0, sigma=10)
#     # BeloningVanWerknemers_8_name = pm.Normal("BeloningVanWerknemers_8", mu=0, sigma=10)
    
#     # M1_name = pm.Normal("M1", mu=0, sigma=10)
#     # AEX_close_name = pm.Normal("AEX_close", mu=0, sigma=10)

#     # random1_name = pm.Normal("random_1", mu=0, sigma=10)
#     # random2_name = pm.Normal("random_2", mu=0, sigma=10)
   
#     sigma = pm.HalfCauchy("sigma", 10)

#     mu = pm.Deterministic("mu", alpha + 
#                           household_cons_name*df1.household_cons 
#                           )
    
#     y_pred = pm.StudentT("y_pred", mu=mu, nu=1, sigma=sigma, observed=df1.gdp_total)

#     idata_one = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
#     idata_one.extend(pm.sample_posterior_predictive(idata_one))

# az.plot_posterior(idata_one, var_names=['~mu', '~alpha'])
# plt.savefig("plot_posterior_small")

# with pm.Model() as model_small:
#     alpha = pm.Normal("alpha", mu=0, sigma=10)
#     household_cons_name = pm.Normal("household_cons", mu=0, sigma=10)
#     Consumentenvertrouwen_1_name = pm.Normal("Consumentenvertrouwen_1", mu=0, sigma=10)
#     DEU_name = pm.Normal("DEU", mu=0, sigma=10)
#     CPI_1_name = pm.Normal("CPI_1", mu=0, sigma=10)
#     M3_2_name = pm.Normal("M3_2", mu=0, sigma=10)
#     AEX_close_name = pm.Normal("AEX_close", mu=0, sigma=10)

#     sigma = pm.HalfCauchy("sigma", 10)

#     mu = pm.Deterministic("mu", alpha + 
#                           household_cons_name*df1.household_cons + 
#                           Consumentenvertrouwen_1_name*df1.Consumentenvertrouwen_1 + 
#                           DEU_name*df1.DEU +
#                           CPI_1_name*df1.CPI_1 + 
#                           M3_2_name*df1.M3_2 +
#                           AEX_close_name*df1.AEX_close
#                           )

#     y_pred = pm.StudentT("y_pred", mu=mu, nu=1, sigma=sigma, observed=df1.gdp_total)
    
#     idata_small = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
#     idata_small.extend(pm.sample_posterior_predictive(idata_small))


# az.plot_posterior(idata_small, var_names=['~mu', '~alpha'])
# plt.savefig("test2")


# ###############################################
# # model comparison
# ###############################################

# _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
# az.plot_ppc(idata_one, num_pp_samples=100, ax=axes[0], legend=False, colors=["C1", "C0", "C1"])
# axes[0].set_title("full linear model")
# az.plot_ppc(idata_small, num_pp_samples=100, ax=axes[1], colors=["C1", "C0", "C1"])
# axes[1].set_title(f"small linear model")

# plt.savefig("comparison1")

# ##################

# fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey="row")
# colors = ["C0", "C1"]
# titles = ["mean", "interquartil range"]
# modelos = ["full model", f"small model"]
# idatas = [idata_one, idata_small]


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

# cmp_df = az.compare({"model_large": idata_one, "model_small": idata_small})
# # cmp_df.to_markdown()
# print(cmp_df)


# ##################

# az.plot_compare(cmp_df)
# plt.savefig("compareplot.png")