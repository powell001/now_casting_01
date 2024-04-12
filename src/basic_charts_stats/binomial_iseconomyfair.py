import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from trend.myhelpers import printme
from scipy import signal
from graphviz import Source
# import preliz as pz #
from scipy.stats import binom
#from utils import pmf_from_dist
from empiricaldist import Pmf


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
df1 = readindata(feat, removeOutlier=False)

def standardize_data(data: pd.DataFrame):
    '''standardize data for plots'''

    df1_std = data.subtract(data.mean()).divide(data.std())
    new_names = [x+"_std" for x in df1_std]
    df1_std.columns = new_names

    return df1_std

df1_std = standardize_data(df1)
df1_std['gdp_total_std_diff'] = df1_std['gdp_total_std'].diff()

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
df1.loc['1996-04-01', 'gdp_total_detrend_positive'] = np.NaN 

df1['gdp_total_std_diff_positive'] = 0
df1['gdp_total_std_diff_positive'][df1['gdp_total_std_diff'] > 0] = 1 
df1.loc['1996-04-01', 'gdp_total_std_diff_positive'] = np.NaN


def binomials():
    
    #############
    # detrend data downturns or upturns?
    rows1 = df1['gdp_total_std_diff_positive'].shape[0]
    upturns = df1['gdp_total_std_diff_positive'].sum()
    downturns = rows1 - df1['gdp_total_std_diff_positive'].sum()
    print(upturns, downturns)

    from scipy.stats import binom

    n = rows1
    p = .5
    k = downturns

    print(binom.pmf(k,n,p))

    #################################
    #################################


    # detrend data downturns or upturns?
    rows1 = df1['gdp_total_detrend_positive'].shape[0]
    upturns = df1['gdp_total_detrend_positive'].sum()
    downturns = rows1 - df1['gdp_total_detrend_positive'].sum()
    print(upturns, downturns)

    from scipy.stats import binom

    n = rows1
    p = .5
    k = downturns

    print(binom.pmf(k,n,p))

binomials()

rows1 = df1.dropna().shape[0]
hypos = np.linspace(0,1,rows1)
prior = Pmf(1, hypos)


likelihood_upturns = hypos
likelihood_downturns = 1-hypos

likelihood = {
    'Up': likelihood_upturns,
    'Down': likelihood_downturns
}

dataset = df1['gdp_total_detrend_positive'].values
dataset = np.where(dataset == 1., "Up", "Down")

def update_gpd(pmf, dataset):
    '''Update pmf with real gdp_total data'''
    for data in dataset:
        pmf *= likelihood[data]

    pmf.normalize()
    return pmf

posterior = prior.copy()
mypmf = update_gpd(posterior, dataset)
max_prob = posterior.max_prob()
plt.plot(mypmf.index, mypmf.iloc[:])
plt.scatter([max_prob], [.00], c = "r")
plt.show()
plt.savefig("fig/gdp_binomial.png")

def plotdata():
    plt.rcParams['axes.grid'] = True
    fig, axs = plt.subplots(6, sharex=True, figsize=(12, 4.5))
    fig.suptitle('Standardize, detrended GDP')
    plt.grid(color='grey', linestyle='-', linewidth=.5)
    axs[0].plot(df1.loc[:, "gdp_total_std"])
    axs[0].title.set_text("gdp_total_std")
    axs[1].plot(df1.loc[:, "gdp_total_std_diff"])
    axs[1].title.set_text("gdp_total_std_diff")
    axs[2].plot(df1.loc[:, "gdp_total_detrend"])
    axs[2].title.set_text("gdp_total_detrend")
    axs[3].plot(df1.loc[:, "gov_invest_std"])
    axs[3].title.set_text("gov_invest_std")
    axs[4].plot(df1.loc[:, "gdp_total_detrend_diff"])
    axs[4].title.set_text("gdp_total_detrend_diff")
    axs[5].plot(df1.loc[:, "gdp_total_detrend_positive"])
    axs[5].title.set_text("gdp_total_detrend_positive")

    plt.savefig("../fig/gdp_allforms.png")
    plt.show()

plotdata()
plt.close()
