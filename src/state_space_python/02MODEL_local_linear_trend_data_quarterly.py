import requests
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from dateutil import relativedelta
from local_linear_trend_code import LocalLinearTrend

data1 = pd.read_csv("data/a0_combinedQuarterly.csv", index_col=[0])

data1.index = pd.date_range(start='4/1/1995', end = '4/1/2024', freq='QS')
print("datat1: ", data1)
_, cols = data1.shape

forecasts = []
allfeatures = []
for i in np.arange(0,cols):
    print("i = : ", i)
    df1 = data1.iloc[25:, [i]]   #25 is arbitrary##########################################################

    # horizon
    horizon = 1

    # Setup the model
    mod = LocalLinearTrend(df1)
    
    # Fit it using MLE (recall that we are fitting the three variance parameters)
    res = mod.fit(disp=False)
    
    # forecast
    fore1 = res.forecast(horizon)
    
    forecasts.append(fore1.values.tolist())


forecasts = [x[0] for x in forecasts]
# mini-dataframe with forecasts
df1 = pd.DataFrame(data=forecasts).T
df1.columns = data1.columns
df1.index = pd.date_range(start='4/1/2024', end = '4/1/2024', freq='QS')
data2 = pd.concat([data1, df1])
data2.to_csv("data/a0_combinedQuarterly_statespace.csv")
data2.to_csv("src/state_space_python/a0_combinedQuarterly_statespace.csv")
