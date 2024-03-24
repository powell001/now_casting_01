# https://www.pymc-labs.com/blog-posts/simulating-data-with-pymc/
import pymc as pm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


x = pm.Gamma.dist(alpha=2, beta=1)
x_draws = pm.draw(x, draws=1000, random_seed=1)
sns.histplot(x_draws);
#plt.show()
plt.close()

x = pm.Gamma.dist(mu=2, sigma=1)
x_draws = pm.draw(x, draws=1000, random_seed=2)
sns.histplot(x_draws);
#plt.show()
plt.close()

x = pm.Dirichlet.dist([[1, 5, 100], [100, 5, 1]])
pm.draw(x, random_seed=3)


x = pm.Truncated.dist(pm.Lognormal.dist(0, 1), upper=3)
x_draws = pm.draw(x, draws=10_000, random_seed=4)
sns.histplot(x_draws);
#plt.show()

x = pm.Mixture.dist(
    w=[0.3, 0.7], 
    comp_dists=[
        pm.Normal.dist(-1, 1), 
        pm.Normal.dist(1, 0.5),
    ],
)
x_draws = pm.draw(x, draws=10_000, random_seed=5)
sns.histplot(x_draws);
#plt.show()


init_dist = pm.Mixture.dist(
    w=[0.3, 0.7], 
    comp_dists=[
        # Why? Because we can!
        pm.Beta.dist(1, 1), 
        pm.Normal.dist(100, 0.5),
    ]
)

x = pm.RandomWalk.dist(
    init_dist=init_dist,
    innovation_dist=pm.StudentT.dist(nu=4, mu=0, sigma=1),
    steps=1000,
)

x_draws = pm.draw(x, draws=5, random_seed=6)
for x_draw in x_draws:
    plt.plot(x_draw)
plt.xlabel("t");
#plt.show()

##########################
##########################
# Multiple non independent variables
##########################
##########################

idx = pm.Categorical.dist(p=[.1, .3, .6])
x = pm.Normal.dist(mu=[-100, 0, 100], sigma=1)[idx]
idx_draws, x_draws = pm.draw([idx, x], draws=5, random_seed=7)

print(idx_draws, x_draws)

n_events = pm.Poisson.dist(5)
emissions = pm.Gamma.dist(mu=10, sigma=2, shape=n_events)
pm.draw([n_events, emissions.sum()], draws=3, random_seed=8)

##########################
##########################
# What if I don't know what I want to sample?
##########################
##########################

df = sns.load_dataset("diamonds")

fg = sns.displot(data=df, x="price", col="cut", facet_kws=dict(sharey=False), height=3.5, aspect=0.85);
for ax in fg.axes.ravel():
    ax.tick_params(axis="both", labelsize=11)

#plt.show()

cut_idxs, cut_labels = pd.factorize(df["cut"])
print(cut_idxs, cut_labels)


coords = {
    "cut": cut_labels.codes,
    "components": (0, 1, 2),
    "obs": range(len(df)),
}
with pm.Model(coords=coords) as m:
    # Priors for the weights, means and standard deviations
    mix_weights = pm.Dirichlet("mix_weights", np.ones((5, 3)), dims=("cut", "components"))
    mix_means = pm.Normal("mix_means", mu=[7, 8, 9], sigma=3, dims=("cut", "components"))
    mix_stds = pm.HalfNormal("mix_stds", sigma=2, dims=("cut", "components"))

    # Distribution of the data
    # We use numpy advanced indexing to broadcast the 5 mixtures parameters 
    # and weights into the long form shape of the data
    price = pm.Mixture(
        "price",
        w=mix_weights[cut_idxs],
        # You can pass a single distribution to Mixture,
        # in which case the last dimensions correspond to the mixture components.
        comp_dists=pm.LogNormal.dist(mu=mix_means[cut_idxs], sigma=mix_stds[cut_idxs]),
        observed=df["price"],
        dims="obs",
    )


#graphvis = pm.model_to_graphviz(m)
#graphvis.view()

with m:
    fit = pm.find_MAP(include_transformed=False)
fit
print(fit)

cut_idxs = cut_labels.codes
price = pm.Mixture.dist(
    w=fit["mix_weights"][cut_idxs],
    comp_dists=pm.LogNormal.dist(
        mu=fit["mix_means"][cut_idxs], 
        sigma=fit["mix_stds"][cut_idxs]
    ),
)
# Each draw returns one price for each cut category
pm.draw(price, random_seed=9)

draws = pm.draw(price, draws=10_000, random_seed=10)

fig, ax = plt.subplots(2, 5, figsize=(16, 6), sharex="col")
for col in range(5):
    sns.histplot(data=df.query(f"cut=='{cut_labels[col]}'"), x="price", binwidth=500, element="step", ax=ax[0, col])
    sns.histplot(data=draws[:, col], binwidth=500, element="step", ax=ax[1, col], color="C1")

    if col == 0:
        ax[0, 0].set_ylabel("Observed data")
        ax[1, 0].set_ylabel("Simulated data");
    else:
        ax[0, col].set_ylabel("")
        ax[1, col].set_ylabel("") 
    ax[0, col].set_title(f"cut={cut_labels[col]}")
    ax[1, col].set_xlabel("price")

for axi in ax.ravel():
    axi.tick_params(axis="both", labelsize=11);

plt.show()


r = sns.displot(data=df, x="price", col="cut", kind="ecdf", log_scale=True, height=3.5, aspect=0.85)
for i in range(5):
    sns.ecdfplot(draws[:, i], ax=r.axes[0, i], color="C1", lw=2)

plt.show()