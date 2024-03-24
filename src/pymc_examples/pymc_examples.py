import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from matplotlib import pyplot as plt


RANDOM_SEED = 8929
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")