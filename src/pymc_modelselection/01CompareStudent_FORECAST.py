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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)

az.style.use("arviz-plasmish")
# from cycler import cycler
# default_cycler = cycler(color=["#6a6a6a", "#bebebe", "#2a2eec",  "#000000"])
# plt.rc("axes", prop_cycle=default_cycler)
plt.rcParams['figure.figsize'] = [15, 7.5]
plt.rcParams['figure.dpi'] = 80

# import preliz as pz #
print(f"Running on PyMC v{pm.__version__}")

mytune = 1000
mydraws = 5000
myn_init = 1000
mychains = 4
mycores = 1

orig1 = pd.read_csv(r"data\a0_combinedQuarterly.csv", index_col=[0])


#########################################################################################
# collect from transformed data, see: "src\\pymc_modelselection\\readinTransform.py" 
#########################################################################################
df1 = collecttransform()

############### DROP NA
df1.dropna(inplace= True)

printme(df1)

df1['gdp_total'].hist();
plt.title("Transformed NL Total GDP")
plt.savefig("fig\gdp_total")

#df1 = df1.iloc[:, [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,46,47]] ############## Select Features
#df1 = df1.iloc[:, [0,1,2,6,7,11,12,18,28,40,41,47]] ############## Select Features
#df1 = df1.iloc[:, [0,1,2,4,9,13,47]] ############## Select Features
#df1 = df1.iloc[:, [0,1,2,4,9,11,12,13,39,40,47,48]]
df1 = df1.iloc[:, [0,1,2,4,9,11,12,13,24,39,40,47]]


X = df1.copy()
y = X.pop("gdp_total")
N, D = X.shape

number1 = 115 ##################### number of observations, may not be N above; N may be less number of observations because N contains 'extended' data 
outsample = df1.shape[0] - number1

# Train ############
X_train = X.iloc[0:number1, :]
y_train = y.iloc[0:number1]

# Test ############
X_test = X.iloc[number1:, :]
y_test = y.iloc[number1:]

# Out sample ############
X_outsample = X.iloc[-outsample:, :]
y_outsample = y.iloc[-outsample:]

################################################################
# Students T
################################################################
with  pm.Model(coords={"predictors": X.columns.values}) as student_model:
    # Prior on error SD
    sigma = pm.HalfNormal("sigma", 1)

    beta = pm.Normal("beta", 0, 1, dims="predictors")

     # No shrinkage on intercept
    alpha = pm.Normal("alpha", 0, 1)

    mu = pm.Deterministic("mu", alpha + at.dot(X.values, beta))

    scores = pm.StudentT("scores", mu=mu, sigma=sigma, nu=2, observed=y.values)

    idata_student = pm.sample(tune = mytune, draws = mydraws, n_init=myn_init, idata_kwargs={"log_likelihood": True}, target_accept=0.98, cores=mycores, chains=mychains)
    idata_student.extend(pm.sample_posterior_predictive(idata_student))

# Compute the point prediction by taking the mean and defining the category via a threshold.
p_test_pred = idata_student.posterior_predictive["scores"].mean(dim=["chain", "draw"])
forecast1 = pd.DataFrame(p_test_pred.values.tolist(), columns=['forecast'])
print(forecast1)

# dffed = forecast1.values.tolist()
# print(dffed)
# print("THIS IS THE CHANGE: ", (dffed * stdGDP) + meanGDP)



# az.plot_posterior(idata_student.posterior_predictive["scores"])
# plt.show()


###################
###################
# Reconstitute
###################
###################

orig1 = pd.read_csv(r"data\a0_combinedQuarterly.csv", index_col=[0])
orig_gdp = orig1['gdp_total']
orig_gdp_diff = orig_gdp.diff()
orig_gdp_mean = orig_gdp_diff.mean()
orig_gdp_std = orig_gdp_diff.std()

d1 = forecast1 * orig_gdp_std + orig_gdp_mean
print(d1)

d1.index = df1.index
print(d1)


#print(df1['gdp_total'])


# reconstitute1 = (df1['gdp_total'] * stdGDP) + meanGDP
# gdp2 = np.append(firstGDP, reconstitute1.dropna())
# print(np.cumsum(gdp2))

# reconstitute1 = (df1['gdp_total'] * stdGDP) + meanGDP
# gdp2 = np.append(firstGDP, reconstitute1.dropna())
# gdp3 = np.append(gdp2, dffed)
# print(np.cumsum(gdp2))

### Final Plot
### transform
# undiff then exp
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

# df1.to_csv("tmp.csv")