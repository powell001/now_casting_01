from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as at
from numpy import random
#from myhelpers import printme
import xarray as xr
#from readinTransform import collecttransform
import pymc_bart as pmb
#detrend
import scipy
#transform data
from readinTransform import collecttransform
#print help
from myhelpers import printme

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)

print(f"Running on PyMC v{pm.__version__}")

RANDOM_SEED = 5781
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")

mytune = 1000
mydraws = 5000
myn_init = 1000
mychains = 4
mycores = 1

#######################
# this is original data
# dt1 = pd.read_csv(r"data\mergedDataforAnalysis_statespace_COMPLETE.csv", index_col=[0])
#######################

#########################################################################################
# collect from transformed data, see: "src\\pymc_modelselection\\readinTransform.py" 
#########################################################################################
dt1 = collecttransform(diff_gdp_total = False)
dt1.to_csv(r"src\pymc_modelselection\tmp_dt1.csv")
dt1.dropna(inplace=True)

dt1_detrend = dt1.copy()
dt1_detrend = dt1.apply(lambda x: scipy.signal.detrend(x, axis=-1, type='linear', bp=0))
dt1_detrend.to_csv(r"src\pymc_modelselection\tmp_detrend.csv")

printme(dt1_detrend)

features = ['lag_gdp_total', 'BusinessOutlook_Industry_monthly', 'Bankruptcies_monthly', 'BusinessOutlook_Retail_monthly', 'Consumentenvertrouwen_1_monthly', 'M1_monthly', 'EA_monthly']

#####################################################
Y = dt1_detrend.loc[:, 'diff_gdp_total'].T
X = dt1_detrend.loc[:, features]
#####################################################

print(X.shape)
print(Y.shape)

if __name__ == "__main__":
    with pm.Model() as bart_g:
        sigma = pm.HalfNormal("sigma", Y.std())
        mu_ = pmb.BART("mu", X, Y, m=50)
        y = pm.Normal("y", mu=mu_, sigma=sigma, observed=Y)
        idata_bart_g = pm.sample(compute_convergence_checks=False, cores=1)

    az.plot_trace(idata_bart_g, var_names=["sigma"], kind="rank_bars");
    plt.show()

    pmb.plot_convergence(idata_bart_g, var_name="mu");
    plt.show()

    pmb.plot_pdp(mu_, X=X, Y=Y, grid=(2, 4));
    plt.show()

    pmb.plot_variable_importance(idata_bart_g, mu_, X);
    plt.show()

    

