import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from myhelpers import printme
# import preliz as pz #
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
az.style.use("arviz-darkgrid")
print(f"Running on PyMC3 v{pm.__version__}")


dfhogg = pd.DataFrame(
    np.array(
        [
            [1, 201, 592, 61, 9, -0.84],
            [2, 244, 401, 25, 4, 0.31],
            [3, 47, 583, 38, 11, 0.64],
            [4, 287, 402, 15, 7, -0.27],
            [5, 203, 495, 21, 5, -0.33],
            [6, 58, 173, 15, 9, 0.67],
            [7, 210, 479, 27, 4, -0.02],
            [8, 202, 504, 14, 4, -0.05],
            [9, 198, 510, 30, 11, -0.84],
            [10, 158, 416, 16, 7, -0.69],
            [11, 165, 393, 14, 5, 0.30],
            [12, 201, 442, 25, 5, -0.46],
            [13, 157, 317, 52, 5, -0.03],
            [14, 131, 311, 16, 6, 0.50],
            [15, 166, 400, 34, 6, 0.73],
            [16, 160, 337, 31, 5, -0.52],
            [17, 186, 423, 42, 9, 0.90],
            [18, 125, 334, 26, 8, 0.40],
            [19, 218, 533, 16, 6, -0.78],
            [20, 146, 344, 22, 5, -0.56],
        ]
    ),
    columns=["id", "x", "y", "sigma_y", "sigma_x", "rho_xy"],
)

dfhogg["id"] = dfhogg["id"].apply(lambda x: "p{}".format(int(x)))
dfhogg.set_index("id", inplace=True)

dfhogg['x_centered'] = (dfhogg.x - dfhogg.x.mean())/(2*dfhogg.x.std())
dfhogg['y_centered'] = (dfhogg.y - dfhogg.y.mean())/(2*dfhogg.y.std())
dfhogg["sigma_x"] = dfhogg["sigma_x"] / (2 * dfhogg["x"].std())
dfhogg["sigma_y"] = dfhogg["sigma_y"] / (2 * dfhogg["y"].std())

print(dfhogg.head())

with pm.Model() as mdl_ols:
    # Define weakly informative Normal priors to give Ridge regression

    x = pm.Data('x', dfhogg['x_centered'].values)
    y = pm.Data('y', dfhogg['y_centered'].values)
    #sigma = pm.Data('sigma', dfhogg['sigma_y'].values)
    
    sigma = pm.HalfNormal("sigma", 100)
    
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    b1 = pm.Normal("b1", mu=0, sigma=10)

    # Define linear model
    y_est = alpha + b1 * x

    # Define Normal likelihood
    y_pred = pm.Normal("y_pred", mu=y_est, sigma=sigma, observed=y)

    idata_trend1 = pm.sample(tune = 5000, draws = 500, n_init=50000, idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
    idata_trend1.extend(pm.sample_posterior_predictive(idata_trend1))

plt.close()
_ = az.plot_trace(idata_trend1, compact=False)
plt.show()


with pm.Model() as mdl_ols:
    # Define weakly informative Normal priors to give Ridge regression

    x = pm.Data('x', dfhogg['x_centered'].values)
    y = pm.Data('y', dfhogg['y_centered'].values)
    
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    b1 = pm.Normal("b1", mu=0, sigma=10)
    sigma = pm.HalfCauchy("sigma", 5)

    # Define linear model
    y_est = alpha + b1 * x

    # define prior for StudentT degrees of freedom
    # InverseGamma has nice properties:
    # it's continuous and has support x ∈ (0, inf)
    nu = pm.InverseGamma("nu", alpha=1, beta=1)

    # Define Normal likelihood
    y_pred = pm.StudentT("y_pred", mu=y_est, nu = nu, sigma=sigma, observed=y)

    idata_student = pm.sample(tune = 5000, draws = 500, n_init=50000, idata_kwargs={"log_likelihood": True}, cores=1, chains=4)
    idata_student.extend(pm.sample_posterior_predictive(idata_student))

plt.close()
_ = az.plot_trace(idata_student, compact=False)
plt.show()