import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from myhelpers import printme
from scipy import signal
from graphviz import Source
# import preliz as pz #


def readindata(features: list, removeOutlier: bool = False):
    dt1 = pd.read_csv(r"data\mergedDataforAnalysis.csv", index_col=[0], parse_dates =[0])

    ########################
    # select columns
    ########################
    dt1 = dt1[features]
  
    ########################
    # reconstitute original gdp data
    ########################
    gdp1 = dt1['gdp_total'].dropna()
    gdp2 = np.append(119388, gdp1)
    gdp3 = np.append(np.cumsum(gdp2), np.NaN)
    dt1['gdp_total'] = gdp3

    if removeOutlier:
        for i in dt1.columns:
            # replace 2020-Q2
            dt1.loc["2020-07-01", i] = dt1.loc["2020-04-01", i]

    print(dt1.describe(include='all'))
    printme(dt1)
 
    return dt1

singlefeature = 'gov_invest'
feat = ['gdp_total', singlefeature]
df1 = readindata(feat, removeOutlier=True)

def standardize_data(data: pd.DataFrame):
    '''standardize data for plots'''

    df1_std = data.subtract(data.mean()).divide(data.std())
    new_names = [x+"_std" for x in df1_std]
    df1_std.columns = new_names

    return df1_std

df1_std = standardize_data(df1)

# merge
df1 = df1.merge(df1_std, left_index=True, right_index=True)

def detrend_data(data: pd.DataFrame):
    '''detrend gdp_total'''

    cols = data.columns
    if "gdp_total_std" in cols:
        gdp_total_detrend = signal.detrend(data.iloc[:-1, :].loc[:, ["gdp_total_std"]], axis=0)
        data['gdp_total_detrend'] = np.append(gdp_total_detrend, np.NaN)
    else:
        print("gdp_total not included in colums")

    return data

df1 = detrend_data(df1)
df1['trend'] = np.arange(0, df1.shape[0])


# create increases/decreases for gdp
df1['gdp_total_detrend_diff'] = df1['gdp_total_detrend'].diff()
df1['gdp_total_detrend_positive'] = 0
df1['gdp_total_detrend_positive'][df1['gdp_total_detrend_diff'] > 0] = 1 

print(df1)

#from utils import pmf_from_dist
#from empiricaldist import Pmf

#############
# downturns or upturns?
rows1 = df1['gdp_total_detrend_positive'].shape[0]
upturns = df1['gdp_total_detrend_positive'].sum()
downturns = rows1 - df1['gdp_total_detrend_positive'].sum()
print(upturns, downturns)

from scipy.stats import binom

n =2
p = .5
k = 1

print(binom.pmf(k,n,p))

# hypos = Pmf.from_seq(np.arange(101).tolist())
# print(hypos)
# prior_dist = Pmf.from_seq(1, hypos)
# print(prior_dist)


def plotdata():
    plt.rcParams['axes.grid'] = True
    fig, axs = plt.subplots(4, sharex=True, figsize=(12, 4.5))
    fig.suptitle('Standardize, detrended GDP')
    plt.grid(color='grey', linestyle='-', linewidth=.5)
    axs[0].plot(df1.loc[:, "gdp_total_detrend"])
    axs[1].plot(df1.loc[:, "gov_invest_std"])
    axs[2].plot(df1.loc[:, "gdp_total_detrend_diff"])
    axs[3].plot(df1.loc[:, "gdp_total_detrend_positive"])

    plt.show()

# plotdata()
# plt.close()

# N = 2000
# with pm.Model() as model_l:
#     # data
#     df1.dropna(inplace = True)
#     x_center = pm.Data('x_center', df1['trend'])
#     y_center = pm.Data('y_center', df1['gdp_total_detrend'])
    
#     α = pm.Normal("α", mu=0, sigma=5)
#     β = pm.Normal("β", mu=0, sigma=10)
#     σ = pm.HalfNormal("σ", 5)

#     μ = pm.Deterministic("μ", α + β*x_center)
    
#     # define likelihood
#     y_pred = pm.Normal("y_pred", mu=μ, sigma=σ, observed= y_center)

#     idata_l = pm.sample(N, idata_kwargs={"log_likelihood": True}, cores=1)
#     idata_l.extend(pm.sample_posterior_predictive(idata_l))

# with pm.Model() as model_cc:
#     # data
#     df1.dropna(inplace = True)
#     x_center = pm.Data('x_center', df1['gov_invest_std'])
#     y_center = pm.Data('y_center', df1['gdp_total_detrend'])
    
#     α = pm.Normal("α", mu=0, sigma=5)
#     β = pm.Normal("β", mu=0, sigma=10)
#     σ = pm.HalfNormal("σ", 5)

#     μ = pm.Deterministic("μ", α + β*x_center)
    
#     # define likelihood
#     y_pred = pm.Normal("y_pred", mu=μ, sigma=σ, observed= y_center)

#     idata_cc = pm.sample(N, idata_kwargs={"log_likelihood": True}, cores=1)
#     idata_cc.extend(pm.sample_posterior_predictive(idata_cc))


# _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
# az.plot_ppc(idata_l, num_pp_samples=100, ax=axes[0], legend=False, colors=["C1", "C0", "C1"])
# axes[0].set_title("linear model")
# az.plot_ppc(idata_cc, num_pp_samples=100, ax=axes[1], colors=["C1", "C0", "C1"])
# axes[1].set_title(f"gov_invest_std")
# plt.show()
# plt.close()   

# # means and quartile
# fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey="row")
# colors = ["C0", "C1"]
# titles = ["mean", "interquartile range"]
# modelos = ["lineal", "other series"]
# idatas = [idata_l, idata_cc]

# #############################
# #############################

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

# plt.show()
# plt.savefig("fig/lin-pol-bpv.png")
# plt.close()

# #############################
# #############################

# fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey="row")
# colors = ["C0", "C1"]
# titles = ["mean", "interquartil range"]
# modelos = ["lineal", "consumer confidence"]
# idatas = [idata_l, idata_cc]

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

# plt.show()
# plt.savefig("fig/lin-pol-bpv.png")
# plt.close()

# #############################
# #############################

# fig, ax = plt.subplots(figsize=(10, 3))

# for idata, c in zip(idatas, colors):
#     az.plot_bpv(idata, color=c, ax=ax)

# plt.show()
# plt.savefig("fig/lin-pol-bpv2.png")        
# plt.close()    

# #############################
# ############################# 

# waic_l = az.waic(idata_l)
# print(waic_l)

# waic_cc = az.waic(idata_cc)
# print(waic_cc)

# loo_l = az.loo(idata_l)
# print(loo_l)

# loo_cc = az.loo(idata_cc)
# print(loo_cc)

# cmp_df = az.compare({"model_l": idata_l, "model_cc": idata_cc})
# # cmp_df.to_markdown()
# print(cmp_df)

# plt.close()
# az.plot_compare(cmp_df)
# plt.savefig("fig/compareplot.png")