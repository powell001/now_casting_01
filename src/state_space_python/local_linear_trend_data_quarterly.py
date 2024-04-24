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
#data1 = data1[['gdp_total']]
data1.index = pd.date_range(start='1995-03-31', end='2024-12-31', freq='QE')
dates = pd.date_range(start='1995-03-31', end='2024-12-31', freq='QE')

emptyDF = pd.DataFrame(np.nan, index=dates,columns = data1.columns )

print(emptyDF)
rows, cols = emptyDF.shape

allfeatures = []
for i in np.arange(0,cols):
    df1 = data1.iloc[:, [i]]
    df1.dropna(inplace=True)
    
    # months to forecast
    delta = relativedelta.relativedelta(data1.index[-1], df1.index[-1], months=3)
    horizon = 3 #delta.months

    # Setup the model
    mod = LocalLinearTrend(df1)
    
    # Fit it using MLE (recall that we are fitting the three variance parameters)
    res = mod.fit(disp=False)

    if horizon == 0:
        fore1 = [np.NaN]
        onefeature = np.append(fore1, df1.values).tolist()
        emptyDF.iloc[-len(onefeature)-1:-1, i] = onefeature
    else:    
        fore1 = res.forecast(horizon)
        onefeature = np.append(df1.values, fore1).tolist()
        emptyDF.iloc[-len(onefeature)-1:-1, i] = onefeature

    
extendedMonthlydata = emptyDF.iloc[:-1,:]
extendedMonthlydata.to_csv("data/a0_combinedQuarterly_statespace.csv")
extendedMonthlydata.to_csv("src/state_space_python/a0_combinedQuarterly_statespace.csv")