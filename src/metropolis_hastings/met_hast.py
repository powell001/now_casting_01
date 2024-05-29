import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# draw X from Normal and put that X into the posterior distribution
# compare to values we are pretty sure are in the good distribution
# but what is a high probability, compare to uniform  
# 1. for each value, sample from the posterior
# 2. compared value of each sample to a 'typical/average' value???
# 3. if there is a high probability and that probability is greater then a value from the uniform distribution
# 4. reset theta to that new value and reset the comparison value

def post(θ, Y, α=1, β=1): 
    if 0 <= θ <= 1: 
        prior = stats.beta(α, β).pdf(θ) 
        like = stats.bernoulli(θ).pmf(Y).prod() 
        prob = like * prior 
    else: 
        prob = -np.inf 
    return prob

Y = stats.bernoulli(0.7).rvs(20) 

n_iters = 1000
can_sd = 0.05 
α = β = 1 
θ = 0.5 
trace = {"θ":np.zeros(n_iters)} 

p2 = post(θ, Y, α, β)

for iter in range(n_iters): 
    θ_can = stats.norm(θ, can_sd).rvs(1) 
    p1 = post(θ_can, Y, α, β) 
    pa = p1/p2 
    
    if pa > stats.uniform(0, 1).rvs(1): 
        θ = θ_can 
        p2 = p1
        
    trace["θ"][iter] = θ 

_, axes = plt.subplots(1,2, sharey=True) 
axes[1].hist(trace["θ"], color="0.5", orientation="horizontal", density=True) 
plt.show()