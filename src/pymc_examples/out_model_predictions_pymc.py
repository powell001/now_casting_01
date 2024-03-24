# https://www.pymc-labs.com/blog-posts/out-of-model-predictions-with-pymc/

# prior predictive: relationship of prior distributions and outcomes, before taking into account data
# sampling: infer the posterior distribution of parameters in the model conditioned on the data
# posterior predictive sampling: can be used to predict new outcomes, conditioned on the posterior parameters

import arviz as az
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

seed = sum(map(ord, "Posterior Predictive"))
rng = np.random.default_rng(seed)
sns.set_style("darkgrid")
sns.set(font_scale=1.3)

print(f"Using arviz version: {az.__version__}")
print(f"Using pymc version: {pm.__version__}")

with pm.Model() as m:
    # y ~ 2 * x
    x = pm.MutableData("x", [-2, -1, 0, 1, 2])
    y_obs = [-4, -1.7, -0.1, 1.8, 4.1]

    # plt.plot([-2, -1, 0, 1, 2], y_obs)
    # plt.show()
    # plt.close()

    beta = pm.Normal("beta")
    y = pm.Normal("y", mu=beta * x, sigma=0.1, shape=x.shape, observed=y_obs)

    idata = pm.sample(random_seed=rng, cores=1, chains = 4)

with m:
    pp = pm.sample_posterior_predictive(idata, random_seed=rng)

# # az.plot_ppc(pp);    
# # plt.show()
# # plt.close()

with m:
    # Make predictions conditioned on new Xs
    pm.set_data({"x": [-1, 3, 5]})
    pp = pm.sample_posterior_predictive(idata, predictions=True, random_seed=rng)

# #az.plot_posterior(pp, group="predictions");
# #plt.show()


with pm.Model() as pred_m:
    # Only x changes
    x = np.array([-1, 0, 1])

    beta = pm.Normal("beta")
    y_pred = pm.Normal("y_pred", mu=beta * x, sigma=0.1, shape=x.shape)

    pp = pm.sample_posterior_predictive(
        idata, 
        var_names=["y_pred"], 
        predictions=True, 
        random_seed=rng,
    )

# # az.plot_posterior(pp, group="predictions");
# # plt.show()

# print(idata.posterior.beta)


with pm.Model() as pred_t_m:
    # Using the same x as in the last example
    x = np.array([-1, 0, 1])

    beta = pm.Normal("beta")

    # Only the likelihood distribution changes
    y_t = pm.StudentT("y_pred_t", nu=4, mu=beta * x, sigma=0.1)

    pp_t = pm.sample_posterior_predictive(
        idata, 
        var_names=["y_pred_t"], 
        predictions=True, 
        random_seed=rng,
    )


# az.plot_posterior(pp, group="predictions");
# az.plot_posterior(pp_t, group="predictions", color="C1");
# plt.show()


with pm.Model() as pred_bern_m:
    x = np.linspace(-1, 1, 25)

    beta = pm.Flat("beta")

    # We again change the functional form of the model
    # Instead of a linear Gaussian we Have a logistic Bernoulli model
    p = pm.Deterministic("p", pm.math.sigmoid(beta * x))
    y = pm.Bernoulli("y", p=p)

    pp = pm.sample_posterior_predictive(
        idata, 
        var_names=["p", "y"], 
        predictions=True, 
        random_seed=rng,
    )

def jitter(x, rng):
    return rng.normal(x, 0.02)

#x = pp.predictions_constant_data["x"]

for i in range(25):
    p = pp.predictions["p"].sel(chain=0, draw=i)
    y = pp.predictions["y"].sel(chain=0, draw=i)

    plt.plot(x, p, color="C0", alpha=.1)
    plt.scatter(jitter(x, rng), jitter(y, rng), s=10, color="k", alpha=.1)

plt.plot([], [], color="C0", label="p")
plt.scatter([], [], color="k", label="y + jitter")
plt.legend(loc=(1.03, 0.75));
plt.show()

##################################
##################################
# Simulating new groups in hierarchical models
##################################
##################################

# Hierarchical models are a powerful class of Bayesian models that allow the back-and-forth flow of information across statistically related groups. 
# One predictive question that arises naturally in such settings, is what to expect from yet unseen groups.

# Think about all the cases where this applies. You may want to predict the lifetime of the next acquired customer, or predict the sales of a new 
# product that has not yet been launched. In both cases, we assume there is some similarity between old and new customers or products.

# The predictions for new schools are informed by the group-level variables mu and tau, which were estimated via sampling of the original subset of 8 schools.

# y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
# sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
# J = 8

# with pm.Model() as eight_schools:
#     eta = pm.Normal("eta", 0, 1, shape=J)

#     # Hierarchical mean and SD
#     mu = pm.Normal("mu", 0, sigma=10)
#     tau = pm.HalfNormal("tau", 10)

#     # Non-centered parameterization of random effect
#     theta = pm.Deterministic("theta", mu + tau * eta)

#     pm.Normal("y", theta, sigma=sigma, observed=y)

#     idata = pm.sample(2000, target_accept=0.9, random_seed=rng, cores=1, chains=4)


# with pm.Model() as ten_schools:
#     # Priors for schools 9 and 10
#     # We assume that the mean of school 10 is expected to be one std above the mean
#     # and have a relatively low measurement error
#     eta_new = pm.Normal("eta_new", mu=[0, 1.0], sigma=1)
#     sigma_new = pm.Uniform("sigma_new", lower=[10, 5], upper=[20, 7])

#     # These are unchanged
#     eta = pm.Normal("eta", 0, 1, shape=J)
#     mu = pm.Normal("mu", 0, sigma=10)
#     tau = pm.HalfNormal("tau", 10)

#     # We concatenate the variables from the old and new groups
#     theta = pm.Deterministic("theta", mu + tau * pm.math.concatenate([eta, eta_new]))
#     pm.Normal("y", theta, sigma=pm.math.concatenate([sigma, sigma_new]))

#     pp = pm.sample_posterior_predictive(idata, var_names=["y"], random_seed=rng)

# print(az.summary(pp, group="posterior_predictive"))

# pps = az.extract(pp, group="posterior_predictive")

# _, ax = plt.subplots(5, 2, figsize=(8, 14), sharex=True, sharey=True)
# for i, axi in enumerate(ax.ravel()):
#     sns.kdeplot(pps["y"][i], fill=True, ax=axi, color="C0" if i < 8 else "C1")
#     axi.axvline(0, ls="--", c="k")
#     axi.set_title(f"School[{i}]")
# plt.tight_layout()
# plt.show()

##################################
##################################
# Forecasting time series
##################################
##################################


mu_true = -0.05
sigma_true = 0.5

y = pm.GaussianRandomWalk.dist(
    init_dist=pm.Normal.dist(), 
    mu=mu_true, 
    sigma=sigma_true,
    steps=99,
)
y_obs = pm.draw(y, random_seed=rng)


plt.title(f"mu={mu_true:.2f}, sigma={sigma_true:.2f}")
plt.plot(y_obs, color="k");
plt.show()

with pm.Model() as m:
    mu = pm.Normal("mu")
    sigma = pm.Normal("sigma")
    y = pm.GaussianRandomWalk(
        "y", 
        init_dist=pm.Normal.dist(), 
        mu=mu, 
        sigma=sigma,
        observed=y_obs
    )

    idata = pm.sample(random_seed=rng, cores=1, chains=4)


# To force a new time series to start where the observations "left off", we define init_dist as a DiracDelta on the last observed y. 
# This will force every predictive series to start at that exact value.

# Note again that the prior distributions don't matter, only the variable names and shapes. We use Flat for sigma as in an earlier example.
# We use Normal for mu because (spoiler alert) we will actually sample from it in the next example.
    

with pm.Model() as forecast_m:
    mu = pm.Normal("mu")

    # Flat sigma for illustration purposes
    sigma = pm.Flat("sigma")

    # init_dist now starts on last observed value of y
    pm.GaussianRandomWalk(
        "y_forecast",
        init_dist=pm.DiracDelta.dist(y_obs[-1]),
        mu=mu,
        sigma=sigma,
        steps=99,
    )

    pp = pm.sample_posterior_predictive(
        idata, 
        var_names=["y_forecast"], 
        predictions=True, 
        random_seed=rng,
    )



steps = np.arange(100, 200)
ax = az.plot_hdi(x=steps, y=pp.predictions["y_forecast"])
# Plot first five forecasts
for i in range(5):
    y = pp.predictions["y_forecast"].isel(chain=0, draw=i)
    ax.plot(steps, y, color="k")
ax.plot(np.arange(100), y_obs, color="k", alpha=0.7)
ax.axvline(100, ls="--", color="k")
ax.set_xticks([50, 150])
ax.set_xticklabels(["observed", "forecast"]);
plt.show()    

##################################
##################################
# Sampling latent variables
##################################
##################################

x_censored_obs = [4.3, 5.0, 5.0, 3.2, 0.7, 5.0]

with pm.Model() as censored_m:
    mu = pm.Normal("mu")
    sigma = pm.HalfNormal("sigma", sigma=1)

    x = pm.Normal.dist(mu, sigma)
    x_censored = pm.Censored(
        "x_censored", 
        dist=x, 
        lower=None, 
        upper=5.0, 
        observed=x_censored_obs,
    )

    idata = pm.sample(random_seed=rng, cores=1, chains=1)

with pm.Model() as uncensored_m:
    mu = pm.Normal("mu")
    sigma = pm.HalfNormal("sigma")

    x = pm.Normal.dist(mu, sigma)
    x_censored = pm.Censored("x_censored", dist=x, lower=None, upper=5.0)

    # This uncensored variable is new
    x_uncensored = pm.Normal("x_uncensored", mu, sigma)

    pp = pm.sample_posterior_predictive(
        idata,
        var_names=["x_censored", "x_uncensored"],
        predictions=True,
        random_seed=rng,
    )


# az.plot_posterior(pp, group="predictions");
