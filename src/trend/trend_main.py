import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from myhelpers import printme
# import preliz as pz #

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

###############################
# basic model
###############################

myN = 10000

N = myN
with pm.Model() as model_trend1:
    # data
    trend_centered = pm.Data('trend_centered', dt1['trend_centered'].values)
    gdp_centered = pm.Data('gdp_centered', dt1['gdp_centered'].values)
    
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    b1 = pm.Normal("b1", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", 5)
  
    mu = pm.Deterministic("mu", alpha + b1*trend_centered )
    
    # define likelihood
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, observed=gdp_centered)

    idata_trend1 = pm.sample(N, idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
    idata_trend1.extend(pm.sample_posterior_predictive(idata_trend1))


with pm.Model() as model_quad1:
    # data
    trend_centered = pm.Data('trend_centered', dt1['trend_centered'].values)
    trend_centered_quad = pm.Data('trend_centered_quad', dt1['trend_centered_quad'].values)
    gdp_centered = pm.Data('gdp_centered', dt1['gdp_centered'].values)
    
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    b1 = pm.Normal("b1", mu=0, sigma=10)
    b2 = pm.Normal("b2", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", 5)
  
    mu = pm.Deterministic("mu", alpha + b1*trend_centered + b2*trend_centered_quad)
    
    # define likelihood
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, observed=gdp_centered)

    idata_quad1 = pm.sample(N, idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
    idata_quad1.extend(pm.sample_posterior_predictive(idata_quad1))    


with pm.Model() as model_trend_student1:
    # data
    trend_centered = pm.Data('trend_centered', dt1['trend_centered'].values)
    gdp_centered = pm.Data('gdp_centered', dt1['gdp_centered'].values)
    
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    b1 = pm.Normal("b1", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", 5)

    # define prior for StudentT degrees of freedom
    # InverseGamma has nice properties:
    # it's continuous and has support x ∈ (0, inf)
    nu = pm.InverseGamma("nu", alpha=1, beta=1)
    
    mu = pm.Deterministic("mu", alpha + b1*trend_centered )
    
    # define likelihood
    y_pred = pm.StudentT("y_pred", mu=mu, sigma=sigma, nu=nu, observed=gdp_centered)

    idata_trend1_robust = pm.sample(N, idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
    idata_trend1_robust.extend(pm.sample_posterior_predictive(idata_trend1_robust))


with pm.Model() as model_trend1_remove_outlier:
    # data
    trend_centered = pm.Data('trend_centered', dt1['trend_centered'].values)
    outlier = pm.Data('outlier', dt1['outlier'].values)

    gdp_centered = pm.Data('gdp_centered', dt1['gdp_centered'].values)
    
    alpha = pm.Normal("alpha", mu=10, sigma=10)
    b1 = pm.Normal("b1", mu=0, sigma=10)
    b2 = pm.Normal("b2", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", 5)
  
    mu = pm.Deterministic("mu", alpha + b1*trend_centered + b2*outlier)
    
    # define likelihood
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, observed=gdp_centered)

    idata_outlier1 = pm.sample(N, idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
    idata_outlier1.extend(pm.sample_posterior_predictive(idata_outlier1))


with pm.Model() as model_trend1_quad_remove_outlier:
    # data
    trend_centered = pm.Data('trend_centered', dt1['trend_centered'].values)
    trend_centered_quad = pm.Data('trend_centered_quad', dt1['trend_centered_quad'].values)
    outlier = pm.Data('outlier', dt1['outlier'].values)

    gdp_centered = pm.Data('gdp_centered', dt1['gdp_centered'].values)
    
    alpha = pm.Normal("alpha", mu=10, sigma=10)
    b1 = pm.Normal("b1", mu=0, sigma=10)
    b2 = pm.Normal("b2", mu=0, sigma=10)
    b3 = pm.Normal("b3", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", 5)
  
    mu = pm.Deterministic("mu", alpha + b1*trend_centered + b2*outlier + b3*trend_centered_quad)
    
    # define likelihood
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, observed=gdp_centered)

    idata_quad_outlier1 = pm.sample(N, idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
    idata_quad_outlier1.extend(pm.sample_posterior_predictive(idata_quad_outlier1))


# Compare distribution with and with robust distribution
_, axes = plt.subplots(5, 1, sharey=True, sharex=True)
az.plot_ppc(idata_trend1, num_pp_samples=100, ax=axes[0], legend=False, colors=["C1", "C0", "C1"])
axes[0].set_title("linear model")
az.plot_ppc(idata_trend1_robust, num_pp_samples=100, ax=axes[1], colors=["C1", "C0", "C1"])
axes[1].set_title(f"linear model Student's t")
az.plot_ppc(idata_quad1, num_pp_samples=100, ax=axes[2], colors=["C1", "C0", "C1"])
axes[2].set_title(f"linear quadradic")
az.plot_ppc(idata_outlier1, num_pp_samples=100, ax=axes[3], colors=["C1", "C0", "C1"])
axes[3].set_title(f"outlier removed")
az.plot_ppc(idata_quad_outlier1, num_pp_samples=100, ax=axes[4], colors=["C1", "C0", "C1"])
axes[4].set_title(f"outlier quad removed")

plt.savefig("src/trend/trend_figs/trend_comparisons.png")

###############################

# means and quartile
fig, axes = plt.subplots(2, 1, sharey="row")
colors = ["C0", "C1", "C2", "C3"]
titles = ["mean", "interquartile range"]

modelos = ["linear trend", "linear trend Student's t", "linear quadradic", "outlier removed", "outlier quad removed"]
idatas = [idata_trend1, idata_trend1_robust, idata_quad1, idata_outlier1, idata_quad_outlier1]

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


########################################

fig, ax = plt.subplots()

for idata, c in zip(idatas, colors):
    az.plot_bpv(idata, color=c, ax=ax)

plt.savefig("src/trend/trend_figs/ypred.png")

######################################
idatas = [idata_trend1, idata_trend1_robust, idata_quad1, idata_outlier1, idata_quad_outlier1]
for i in idatas:
    waic_l = az.waic(i)
    print(waic_l)

for i in idatas:
    loo_l = az.loo(i)
    print(loo_l)

cmp_df = az.compare({"trend1": idata_trend1, "trend1_student": idata_trend1_robust, "trend_quad": idata_quad1, "trend_outlier": idata_outlier1, "trend_quad_outlier": idata_quad_outlier1})
print(cmp_df)
cmp_df.to_csv("model_comparison_table.csv")


