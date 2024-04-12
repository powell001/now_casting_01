#https://discourse.pymc.io/t/pymc-experimental-now-includes-state-spaces-models/12773
#https://github.com/pymc-devs/pymc-experimental/blob/main/notebooks/SARMA%20Example.ipynb


import sys
sys.path.append("..")
import jax
jax.config.update("jax_platform_name", "cpu")
import numpyro
numpyro.set_host_device_count(4)

import pymc as pm
import pytensor
from pytensor import tensor as pt

import arviz as az
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import pymc_experimental.statespace as pmss

config = {
    "figure.figsize": [12.0, 4.0],
    "figure.dpi": 72.0 * 2,
    "figure.facecolor": "w",
    "figure.constrained_layout.use": True,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "axes.spines.left": False,
    "axes.spines.right": False,
}

plt.rcParams.update(config)


################
# data
################

seed = sum(map(ord, "statespace arima"))
rng = np.random.default_rng(seed)

AR_params = [0.8, 0.0]
MA_params = [-0.5]

# Initial state
x0 = np.r_[[0.0], [0.0]]

# Hidden state transition matrix
T = np.array([[AR_params[0], 1.0], [AR_params[1], 0.0]])

# Hidden state noise coefficients
R = np.array([[1.0], [MA_params[0]]])

# Hidden state covaraince matrix
Q = np.array([[0.8]])

# Observation matrix
Z = np.array([[1.0, 0.0]])

# Observation noise covariance
H = np.array([[0.0]])

timesteps = 100
data = np.zeros(timesteps)
hidden_states = np.zeros((timesteps, 2))
hidden_states[0, :] = x0

innovations = rng.multivariate_normal(mean=np.array([0.0]), cov=Q, size=timesteps)

for t in range(1, timesteps):
    hidden_states[t] = T @ hidden_states[t - 1, :] + R @ np.atleast_1d(innovations[t])
    data[t] = Z @ hidden_states[t]

fake_dates = pd.date_range("2010-01-01", freq="MS", periods=data.shape[0])
df = pd.DataFrame(data, columns=["state"], index=fake_dates)

fig, ax = plt.subplots(figsize=(14, 4), dpi=100)
df.plot(ax=ax)
plt.show()

mod = sm.tsa.SARIMAX(endog=data, order=(1, 0, 1))
res = mod.fit(disp=0)
print(res.summary())

import pymc_experimental.statespace as pmss
ss_mod = pmss.BayesianSARIMA(order=(1, 1, 1), stationary_initialization=False)


with pm.Model(coords=ss_mod.coords) as arma_model:
    state_sigmas = pm.Gamma("sigma_state", alpha=10, beta=2, dims=ss_mod.param_dims["sigma_state"])
    rho = pm.Beta("ar_params", alpha=5, beta=1, dims=ss_mod.param_dims["ar_params"])
    theta = pm.Normal("ma_params", mu=0.0, sigma=0.5, dims=ss_mod.param_dims["ma_params"])

    ss_mod.build_statespace_graph(df, mode="JAX")
    prior = pm.sample_prior_predictive(compile_kwargs={"mode": "JAX"})


ss_mod.coords