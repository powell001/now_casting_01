import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from myhelpers import printme
from scipy import signal
from graphviz import Source
# import preliz as pz #

df1 = pd.read_csv(r"data\mergedDataforAnalysis.csv", index_col=[0], parse_dates =[0])

columns = ['gdp_total', 'BeloningSeizoengecorrigeerd_2', 'Bankruptcies', 'Consumentenvertrouwen_1']

df2 = df1[columns]
print(df2)
############
# log gdp
############

df2.iloc[1:-1].loc[:,'gdp_total'] = np.log(df2.iloc[1:-1].loc[:, ['gdp_total']].values)
df2.dropna(inplace=True)

def standardize_data(data: pd.DataFrame):
    '''standardize data for plots'''

    df1_std = data.subtract(data.mean()).divide(data.std())
    new_names = [x+"_std" for x in df1_std]
    df1_std.columns = new_names

    return df1_std

df2_std = standardize_data(df2)
print(df2_std)

y_data = df2_std['gdp_total_std']

x1 = df2_std.iloc[:,1]
x2 = df2_std.iloc[:,2]
x3 = df2_std.iloc[:,3]


with pm.Model() as model1:
    b0 = pm.Uniform('b0', -4, 4)
    b1 = pm.Uniform('b1', -4, 4)
    b2 = pm.Uniform('b2', -4, 4)
    b3 = pm.Uniform('b3', -4, 4)

    sigma = pm.Uniform('sigma', 0, 2)
    y_est = b0 + b1*x1 + b2*x2 + b3*x3

    y = pm.Normal('y',
                  mu = y_est, sigma = sigma,
                  observed = y_data
                  )
    
with model1:
    trace1 = pm.sample(2000, idata_kwargs={"log_likelihood": True}, cores = 1)
    trace1.extend(pm.sample_posterior_predictive(trace1))


param_names = ['b1', 'b2', 'b3']
print(trace1.posterior['b1'].mean())

means = [trace1.posterior[name].mean() for name in param_names]
print(means)

def credible_interval(sample):
    ci = np.percentile(sample, [3,97])
    return np.round(ci,3)

cls = [credible_interval(trace1.posterior[name]) for name in param_names]
print(cls)