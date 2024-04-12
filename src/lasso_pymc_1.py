import sys
print(sys.prefix)
import warnings
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr
warnings.simplefilter(action='ignore', category=FutureWarning)
print(f"Running on PyMC3 v{pm.__version__}")

######### Graphics ##########
az.style.use("arviz-grayscale")
from cycler import cycler

cmap = mpl.colormaps['gray']
gray_cycler = cycler(color=cmap(np.linspace(0, 0.9, 6)))

default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
plt.rc("axes", prop_cycle=default_cycler)
plt.rc("figure", dpi=120)
np.random.seed(123)
######### Graphics End ##########

######### Data ##########
df1 = pd.read_csv("lasso_data.csv", usecols=np.arange(start=1, stop=7))
print(df1.head())
df1 = (df1 - df1.mean())/df1.std()
print(df1.head())
######### Data End ##########

# with pm.Model() as model_lb:
#     alpha = pm.Normal("alpha", mu=0, sigma=100)
#     lotarea = pm.Normal("lotarea", mu=0, sigma=10)
#     grlivarea = pm.Normal("grlivarea", mu=0, sigma=10)
#     totrmsabvgrd = pm.Normal("totrmsabvgrd", mu=0, sigma=10)
#     overallqual = pm.Normal("overallqual", mu=0, sigma=10)
#     garagearea = pm.Normal("garagearea", mu=0, sigma=10)
   
#     sigma = pm.HalfCauchy("sigma", 10)

#     mu = pm.Deterministic("mu", alpha + lotarea*df1.LotArea + grlivarea*df1.GrLivArea + totrmsabvgrd*df1.TotRmsAbvGrd + overallqual*df1.OverallQual + garagearea*df1.GarageArea)
#     y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, observed=df1.Price)

#     idata_lb = pm.sample(2000, idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
#     idata_lb.extend(pm.sample_posterior_predictive(idata_lb))

# with pm.Model() as model_lb_small:
#     alpha = pm.Normal("alpha", mu=0, sigma=100)
#     lotarea = pm.Normal("lotarea", mu=0, sigma=10)
#     overallqual = pm.Normal("overallqual", mu=0, sigma=10)
   
#     sigma = pm.HalfCauchy("sigma", 10)

#     mu = pm.Deterministic("mu", alpha + lotarea*df1.LotArea + overallqual*df1.OverallQual)
#     y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, observed=df1.Price)
    
#     idata_lb_small = pm.sample(2000, idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
#     idata_lb_small.extend(pm.sample_posterior_predictive(idata_lb_small))


# # fmt: off
# az.plot_posterior(idata_lb, var_names=["alpha", "lotarea", "grlivarea", "totrmsabvgrd", "overallqual", "garagearea"])
# plt.savefig("test-4")

# idata_lb.posterior["lotarea"] = idata_lb.posterior["lotarea"] * df1.LotArea.std()
# idata_lb.posterior["grlivarea"] = idata_lb.posterior["grlivarea"] * df1.GrLivArea.std()
# idata_lb.posterior["totrmsabvgrd"] = idata_lb.posterior["totrmsabvgrd"] * df1.TotRmsAbvGrd.std()
# idata_lb.posterior["overallqual"] = idata_lb.posterior["overallqual"] * df1.OverallQual.std()
# idata_lb.posterior["garagearea"] = idata_lb.posterior["garagearea"] * df1.GarageArea.std()

# az.plot_forest([idata_lb], model_names=["model_lb"],
# var_names=["lotarea", "grlivarea","totrmsabvgrd", "overallqual","garagearea"], combined=True)
# plt.savefig("test-3")

# az.plot_trace(idata_lb, var_names= ["lotarea", "grlivarea","totrmsabvgrd", "overallqual","garagearea"], combined=True)
# plt.savefig("test-2")

# #######################################

# # fmt: off
# az.plot_posterior(idata_lb_small, var_names=["alpha", "lotarea", "overallqual"], figsize=(12, 3))
# plt.savefig("test-1")

# idata_lb_small.posterior["lotarea"] = idata_lb_small.posterior["lotarea"] * df1.LotArea.std()
# idata_lb.posterior["overallqual"] = idata_lb.posterior["overallqual"] * df1.OverallQual.std()

# az.plot_forest([idata_lb], model_names=["model_lb_small"],
# var_names=["lotarea", "overallqual"], combined=True)
# plt.savefig("test0")

# az.plot_trace(idata_lb, var_names= ["lotarea","overallqual"], combined=True)
# plt.savefig("test1")

# ###############################################
# # model comparison
# ###############################################

# _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
# az.plot_ppc(idata_lb, num_pp_samples=100, ax=axes[0], legend=False, colors=["C1", "C0", "C1"])
# axes[0].set_title("full linear model")
# az.plot_ppc(idata_lb_small, num_pp_samples=100, ax=axes[1], colors=["C1", "C0", "C1"])
# axes[1].set_title(f"small linear model")

# plt.savefig("test2")


# ###############################################
# # model comparison 2
# ###############################################

# fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey="row")
# colors = ["C0", "C1"]
# titles = ["mean", "interquartil range"]
# modelos = ["full model", "subset model"]
# idatas = [idata_lb, idata_lb_small]

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

# plt.savefig("test3")

# fig, ax = plt.subplots(figsize=(10, 3))

# for idata, c in zip(idatas, colors):
#     az.plot_bpv(idata, color=c, ax=ax)

# plt.savefig("test4")


# ###############################################
# # model comparison 3
# ###############################################

# waic_lb = az.waic(idata_lb)
# print(waic_lb)

# waic_lb_small = az.waic(idata_lb_small)
# print(waic_lb_small)

# loo_lb = az.loo(idata_lb)
# print(loo_lb)

# loo_lb_small = az.loo(idata_lb_small)
# print(loo_lb_small)

# cmp_df = az.compare({"model_lb": idata_lb, "model_lb_small": idata_lb_small})
# # cmp_df.to_markdown()
# print(cmp_df)

# # with pm.Model() as mdl_ols:
# #     ## define Normal priors to give Ridge regression
# #     b0 = pm.Normal("Intercept", mu=0, sigma=100)
# #     b1 = pm.Normal("x", mu=0, sigma=100)

# #     ## define Linear model
# #     yest = b0 + b1 * df_lin["x"]

# #     ## define Normal likelihood with HalfCauchy noise (fat tails, equiv to HalfT 1DoF)
# #     y_sigma = pm.HalfCauchy("y_sigma", beta=10)
# #     likelihood = pm.Normal("likelihood", mu=yest, sigma=y_sigma, observed=df_lin["y"])

# #     idata_ols = pm.sample(2000, return_inferencedata=True)