# Compare Normal and StudentsT distribution using graphs and table summaries.

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from myhelpers import printme
import xarray as xr
# import preliz as pz #
plt.rcParams['figure.figsize'] = (12, 8)

print(f"Running on PyMC3 v{pm.__version__}")

### plots ###
az.style.use("arviz-grayscale")
from cycler import cycler
default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
plt.rc("axes", prop_cycle=default_cycler)
plt.rc("figure", dpi=100)
###

### take a look ###
dt1 = pd.read_csv(r"data/gdp_eurostatMar7th.csv", index_col=[0])
dt1['NL'].plot()
plt.savefig("src/trend/trend_figs/gdp.png")

### transform data ###
dt1 = dt1[['NL']]
dt1.dropna(inplace=True)
dt1.rename(columns={"NL": "gdp"}, inplace=True)
dt1['trend'] = np.arange(dt1.shape[0])
tmp = (dt1 - dt1.mean())/dt1.std()
dt1 = dt1.merge(tmp, left_index=True, right_index=True)

dt1.columns = ['gdp','trend', 'gdp_centered', 'trend_centered']
dt1['trend_centered_quad'] = dt1['trend_centered']*dt1['trend_centered']

dt1['outlier'] = 0
dt1['outlier'].iloc[dt1.index == "2020-Q2 "] = 1

printme(dt1)

dt1[['gdp_centered', 'trend_centered']].plot()
plt.savefig("src/trend/trend_figs/gdp_centered.png")

# ### boxplots ##################1

# _, ax = plt.subplots()
# ax.boxplot(dt1['gdp_centered'], vert=False)
# plt.savefig("src/trend/trend_figs/boxplot_gdp_centered.png")

# _, ax = plt.subplots()
# ax.hist(dt1['gdp_centered'], bins = 20)
# plt.savefig("src/trend/trend_figs/hist_gdp_centered.png")
# plt.close()

# ###############################
# # basic model
# ###############################

mytune = 5000
mydraws = 500
myn_init = 50000


with pm.Model() as model_trend1:
    # data
    trend_centered = pm.Data('trend_centered', dt1['trend_centered'].values, mutable=True)
    gdp_centered = pm.Data('gdp_centered', dt1['gdp_centered'].values, mutable=True)
    
    alpha = pm.Normal("alpha", mu=0, sigma=100)
    b1 = pm.Normal("b1", mu=0, sigma=100)
    # b2 = pm.Normal("b2", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", 10)
  
    mu = pm.Deterministic("mu", alpha + b1*trend_centered)
 
    # define likelihood
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, observed=gdp_centered)

    idata_trend1 = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
    idata_trend1.extend(pm.sample_posterior_predictive(idata_trend1))


az.plot_posterior(idata_trend1, var_names=["alpha", "b1","sigma"])
plt.savefig("src/trend/trend_figs/idata_trend1.png")

graphvis = pm.model_to_graphviz(model_trend1)
graphvis.view()


with pm.Model() as idata_trend1_robust:
    # data
    trend_centered = pm.Data('trend_centered', dt1['trend_centered'].values)
    gdp_centered = pm.Data('gdp_centered', dt1['gdp_centered'].values)
    
    alpha = pm.Normal("alpha", mu=0, sigma=100)
    b1 = pm.Normal("b1", mu=0, sigma=100)
    sigma = pm.HalfNormal("sigma", 50)

    # define prior for StudentT degrees of freedom; InverseGamma has nice properties: it's continuous and has support x ∈ (0, inf)
    #nu = pm.InverseGamma("nu", alpha=1, beta=1)
    nu = pm.Exponential('nu', 1./10, testval = 5.)

    
    mu = pm.Deterministic("mu", alpha + b1*trend_centered)
    
    # define likelihood
    y_pred = pm.StudentT("y_pred", mu=mu, sigma=sigma, nu=3, observed=gdp_centered)

    idata_trend1_robust = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
    idata_trend1_robust.extend(pm.sample_posterior_predictive(idata_trend1_robust))

az.plot_posterior(idata_trend1_robust, var_names=["alpha", "b1", "sigma"])
plt.savefig("src/trend/trend_figs/idata_trend1_robust.png")

# Compare distribution with and with robust distribution
_, axes = plt.subplots(2, 1, sharey=True, sharex=True)
az.plot_ppc(idata_trend1, num_pp_samples=100, ax=axes[0], legend=False, colors=["C1", "C0", "C1"])
axes[0].set_title("linear model")
az.plot_ppc(idata_trend1_robust, num_pp_samples=100, ax=axes[1], colors=["C1", "C0", "C1"])
axes[1].set_title(f"linear model Student's t")
plt.savefig("src/trend/trend_figs/trend_comparisons.png")

################################
posterior = az.extract(idata_trend1_robust, num_samples=50)
x_plot = xr.DataArray(
    np.linspace(dt1['trend_centered'].min(), dt1['trend_centered'].max(), 50), dims="plot_id")
mean_line = posterior["alpha"].mean() + posterior["b1"].mean() * x_plot
lines = posterior["alpha"] + posterior["b1"] * x_plot
hdi_lines = az.hdi(idata_trend1_robust.posterior["mu"])

fig, axes = plt.subplots(1, 2, sharey=True)
axes[0].plot(dt1['trend_centered'], dt1['gdp_centered'], "C2.", zorder=-3)
lines_ = axes[0].plot(x_plot, lines.T, c="C1", alpha=0.2, label="lines")
plt.setp(lines_[1:], label="_")
axes[0].plot(x_plot, mean_line, c="C0", label="mean line")
axes[0].set_xlabel("trend robust")
axes[0].set_ylabel("gdp centered")
axes[0].legend()
axes[1].plot(dt1['trend_centered'], dt1['gdp_centered'], "C2.", zorder=-3)
idx = np.argsort(dt1['trend_centered'].values)
axes[1].fill_between(
    dt1['trend_centered'][idx],
    hdi_lines["mu"][:, 0][idx],
    hdi_lines["mu"][:, 1][idx],
    color="C1",
    label="HDI",
    alpha=0.5,
)
axes[1].plot(x_plot, mean_line, c="C0", label="mean line")
axes[1].set_xlabel("trend centered")
axes[1].legend()
plt.savefig("src/trend/trend_figs/insampleest.png")


# ###############################

# means and quartile
fig, axes = plt.subplots(2, 1, sharey="row")
colors = ["C0", "C1", "C2", "C3"]
titles = ["mean", "interquartile range"]

modelos = ["linear trend", "linear trend Student's t"]
idatas = [idata_trend1, idata_trend1_robust]

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

plt.savefig("src/trend/trend_figs/interquartile_range.png")


# ########################################

fig, ax = plt.subplots()

for idata, c in zip(idatas, colors):
    az.plot_bpv(idata, color=c, ax=ax)

plt.savefig("src/trend/trend_figs/ypred.png")

######################################
idatas = [idata_trend1, idata_trend1_robust]
for i in idatas:
    waic_l = az.waic(i)
    print(waic_l)

for i in idatas:
    loo_l = az.loo(i)
    print(loo_l)

cmp_df = az.compare({"trend1": idata_trend1, "trend_studentsT": idata_trend1_robust})
print(cmp_df)
cmp_df.to_csv("model_comparison_table.csv")

######################################

az.plot_compare(cmp_df)
plt.savefig("src/trend/trend_figs/compareplot.png")

######################################

idata_w = az.weight_predictions(idatas, weights=[0.35, 0.65])
_, ax = plt.subplots()
az.plot_kde(
    idata_trend1.posterior_predictive["y_pred"].values,
    plot_kwargs={"color": "C0", "lw": 3},
    label="linear",
    ax=ax,
)
az.plot_kde(
    idata_trend1_robust.posterior_predictive["y_pred"].values,
    plot_kwargs={"color": "C1", "lw": 3},
    label="quadratic",
    ax=ax,
)
az.plot_kde(
    idata_w.posterior_predictive["y_pred"].values,
    plot_kwargs={"color": "C2", "lw": 3, "ls": "--"},
    label="weighted",
    ax=ax,
)

plt.legend()
plt.savefig("src/trend/trend_figs/lin-pol-weighted.png")

az.plot_trace(idata_trend1, var_names=["alpha", "b1","sigma"])
plt.savefig("src/trend/trend_figs/idata_trace_trend.png")

az.plot_trace(idata_trend1_robust, var_names=["alpha", "b1","sigma"])
plt.savefig("src/trend/trend_figs/idata_trace_studentst.png")

az.summary(idata_trend1, kind="stats").round(2)
az.summary(idata_trend1_robust, kind="stats").round(2)
