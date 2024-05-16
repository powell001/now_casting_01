import pandas as pd
import os
import warnings
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
import xarray as xr
import seaborn as sns
import sys
sys.path.append(r'C:/Users/jpark/VSCode/now_casting_01/src/')
from myhelpers import printme
warnings.filterwarnings("ignore", module="scipy")
print(f"Running on PyMC v{pm.__version__}")

from scipy.special import expit as inverse_logit
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
az.style.use("arviz-whitegrid")


# ideas: is there a pattern in the rising and falling of Dutch GDP
# rise/fall data
dt1 = pd.read_csv(r"data/standarized_detrended_gdp_other.csv", index_col=[0])
dt1.index = pd.to_datetime(dt1.index)
# all other data
dt2 = pd.read_csv(r"data/mergedDataforAnalysis.csv", index_col=[0])
dt2.index = pd.to_datetime(dt2.index)
#combine
df1 = dt1.merge(dt2, left_index=True, right_index=True, how="outer")

# poison using bayesian approach
# https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-out-of-sample-predictions.html

df = df1[['gdp_total_detrend_positive', 'Consumentenvertrouwen_1','G20', 'trend', 'CPIAfgeleid_2', 'CHN', 'M1']]
# pairs plot
def pairsplot():
    sns.pairplot(data=df, kind="scatter")
    plt.show()
    plt.close()

y = df.pop('gdp_total_detrend_positive')
print("Naive y percent: ", y.mean())

df1 = (df - df.mean())/df.std()

def scatterplot():
    fig, ax = plt.subplots()
    sns.scatterplot(x=df1["Consumentenvertrouwen_1"], y=df1["G20"], hue=y)
    ax.legend(title="y")
    ax.set(title="Sample Data", xlim=(-9, 9), ylim=(-9, 9));
    plt.show()

df1['intercept'] = np.ones(len(df))

# get order right
labels = ['intercept', 'Consumentenvertrouwen_1', 'G20', 'trend', 'CPIAfgeleid_2', 'CHN', 'M1']
df1 = df1[labels]

# split train test
train_index = np.arange(0,90).tolist()
test_index = np.arange(90, 114).tolist()

x_train, x_test = df1.iloc[train_index, :], df1.iloc[test_index, :]
y_train, y_test = y[train_index], y[test_index]

coords = {"coeffs": labels}

with pm.Model(coords=coords) as model:
    # data containers
    X = pm.MutableData("X", x_train)
    y = pm.MutableData("y", y_train)
    # priors
    b = pm.Normal("b", mu=0, sigma=1, dims="coeffs")
    # linear model
    mu = pm.math.dot(X, b)
    # link function
    p = pm.Deterministic("p", pm.math.invlogit(mu))
    # likelihood
    pm.Bernoulli("obs", p=p, observed=y)

def mygraphvis():
    pm.model_to_graphviz(model)
    graphvis = pm.model_to_graphviz(model)
    graphvis.view()
    plt.show()
    plt.close()
# mygraphvis

with model:
    idata = pm.sample(draws = 4000, idata_kwargs={"log_likelihood": True}, target_accept=0.98, cores=1, chains=4)

def plot_trace():
    az.plot_trace(idata, var_names="b", compact=False);
    plt.show()
    plt.close()
#plot_trace()

print(az.summary(idata, var_names="b"))

def plot_posterior():
    az.plot_posterior(idata, var_names=["b"], figsize=(15, 4));
    plt.show()
    plt.close()
plot_posterior()

with model:
    pm.set_data({"X": x_test, "y": y_test})
    idata.extend(pm.sample_posterior_predictive(idata))

# Compute the point prediction by taking the mean and defining the category via a threshold.
p_test_pred = idata.posterior_predictive["obs"].mean(dim=["chain", "draw"])
y_test_pred = (p_test_pred >= 0.5).astype("int").to_numpy()
print(p_test_pred)
print(y_test_pred)

print(f"accuracy = {np.mean(y_test==y_test_pred): 0.3f}")

def roc():
    fpr, tpr, thresholds = roc_curve(
        y_true=y_test, y_score=p_test_pred, pos_label=1, drop_intermediate=False
    )
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    roc_display = roc_display.plot(ax=ax, marker="o", markersize=4)
    ax.set(title="ROC");
    plt.show()

roc()

# def make_grid():
#     x1_grid = np.linspace(start=-9, stop=9, num=300)
#     x2_grid = x1_grid
#     x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
#     x_grid = np.stack(arrays=[x1_mesh.flatten(), x2_mesh.flatten()], axis=1)
#     return x1_grid, x2_grid, x_grid


# x1_grid, x2_grid, x_grid = make_grid()

# with model:
#     # Create features on the grid.
#     x_grid_ext = np.hstack(
#         (
#             np.ones((x_grid.shape[0], 1)),
#             x_grid,
#             (x_grid[:, 0] * x_grid[:, 1]).reshape(-1, 1),
#         )
#     )
#     # set the observed variables
#     pm.set_data({"X": x_grid_ext})
#     # calculate pushforward values of `p`
#     ppc_grid = pm.sample_posterior_predictive(idata, var_names=["p"])


# # grid of predictions
# grid_df = pd.DataFrame(x_grid, columns=["Consumentenvertrouwen_1", "G20"])
# grid_df["p"] = ppc_grid.posterior_predictive.p.mean(dim=["chain", "draw"])
# p_grid = grid_df.pivot(index="G20", columns="Consumentenvertrouwen_1", values="p").to_numpy()

# def calc_decision_boundary(idata, x1_grid):
#     # posterior mean of coefficients
#     intercept = idata.posterior["b"].sel(coeffs="Intercept").mean().data
#     b1 = idata.posterior["b"].sel(coeffs="x1").mean().data
#     b2 = idata.posterior["b"].sel(coeffs="x2").mean().data
#     b1b2 = idata.posterior["b"].sel(coeffs="x1:x2").mean().data
#     # decision boundary equation
#     return -(intercept + b1 * x1_grid) / (b2 + b1b2 * x1_grid)


# fig, ax = plt.subplots()

# # data
# sns.scatterplot(
#     x=x_test[:, 1].flatten(),
#     y=x_test[:, 2].flatten(),
#     hue=y_test,
#     ax=ax,
# )

# # decision boundary
# ax.plot(x1_grid, calc_decision_boundary(idata, x1_grid), color="black", linestyle=":")

# # grid of predictions
# ax.contourf(x1_grid, x2_grid, p_grid, alpha=0.3)

# ax.legend(title="y", loc="center left", bbox_to_anchor=(1, 0.5))
# ax.set(title="Model Decision Boundary", xlim=(-9, 9), ylim=(-9, 9), xlabel="Consumentenvertrouwen_1", ylabel="G20");




# coords = {"coeffs": labels}
# with pm.Model(coords=coords) as model:
#     # data containers
#     X = pm.MutableData("X", x_train)
#     y = pm.MutableData("y", y_train)
#     # priors
#     b = pm.Normal("b", mu=0, sigma=1, dims="coeffs")
#     # linear model
#     mu = pm.math.dot(X, b)
#     # link function
#     p = pm.Deterministic("p", pm.math.invlogit(mu))
#     # likelihood
#     pm.Bernoulli("obs", p=p, observed=y)

# pm.model_to_graphviz(model)



# y = dt1['gdp_total_detrend_positive']
# y_fake = np.repeat([1,0], [50,50])

# with pm.Model() as model1:
#     theta = pm.Beta("theta", alpha=50, beta=50)

#     y_obs = pm.Binomial("y_obs", n=1, p=theta, observed=y)

#     idata = pm.sample(2000, return_inferencedata=True, cores=1, chains=2, idata_kwargs={'log_likelihood':True}) 

#     idata.extend(pm.sample_prior_predictive(8000))

# print(idata)

# print(idata.posterior)
# print(idata.log_likelihood)
# print(idata.observed_data)
# print(idata.sample_stats)

# # az.plot_bf(idata, var_name = 'theta', ref_val=0.50)
# # plt.show()

# trace = idata['prior_predictive']
# sample_prior = trace['y_obs']
# print(sample_prior)
# az.plot(sample_prior)
# plt.show()



# # ref value, are downturns as likely as upturns. Theta of 0.5 represents no bias   
# # the graph below shows that there are probably more upturns than downturns.
# # How can that be?  One explanation is that downturns tend to be sharper and shorter than
# # upturns. The KDE for the prior and posterior.
# # Bayes factor for BF_01 is 0.98, Using Harold Jeffreys' proposed scale, the result
# # provides no support for the assumption that upturns and downturns are equally
# # likely to be realized.  The method is sensitive to priors


# # comparing models, Martin, Chapter 5
# # from scipy.special import betaln

# # def beta_binom(prior, y):
# #     alpha, beta = prior
# #     h = np.sum(y)
# #     n = len(y)
# #     p_y = np.exp(betaln(alpha + h, beta + n - h) - betaln(alpha, beta))

# #     return p_y

# # #y = np.repeat([1,0], [50,50])
# # priors = ((1,1), (5,5))

# # BF = beta_binom(priors[1], Y)/beta_binom(priors[0],Y)
# # print(round(BF))

# # # comparing models, Martin, Chapter 5
# # models = []
# # idatas = []
# # for alpha, beta in priors:
# #     with pm.Model() as model:
# #         a = pm.Beta("a", alpha, beta)
# #         y1 = pm.Bernoulli("y1", a, observed=Y)
# #         idata = pm.sample_smc(random_seed=42, return_inferencedata=True, cores=1, chains=2, idata_kwargs={'log_likelihood':True})
# #         models.append(model)
# #         idatas.append(idata)

# # BF_smc = np.exp(idatas[1].sample_stats['log_marginal_likelihood'].mean() - idatas[0].sample_stats['log_marginal_likelihood'].mean())

# # print(np.round(BF_smc).item())


# # Bayes factors and inferences

