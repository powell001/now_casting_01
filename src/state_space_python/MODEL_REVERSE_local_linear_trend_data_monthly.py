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

data1 = pd.read_csv("src\state_space_python\mergedDataforAnalysis_statespace.csv", index_col=[0])
data1.index = pd.date_range(start='4/1/1995', end = '7/1/2024', freq='QS')

print("datat1: ", data1)
_, cols = data1.shape

forecasts = []
allfeatures = []
for i in np.arange(0,cols):
    print("i = : ", i)
    df1 = data1.iloc[:-1, [i]]   # start at 1

    # horizon
    horizon =  df1.isna().sum().values[0]
    print("horizon: ", horizon)

    if horizon > 0:
        # create new, temporary dataframe to put nans on end of dataframe
        hz1 = df1.values[horizon:].tolist()
        hz2 = np.append(hz1[::-1], np.repeat(np.nan, horizon))    
        df2 = pd.DataFrame(hz2)
        df2.index = df1.index
    
        # Setup the model
        mod = LocalLinearTrend(df2.iloc[:-horizon,:])
        
        # Fit it using MLE (recall that we are fitting the three variance parameters)
        res = mod.fit(disp=False)
        
        # forecast
        fore1 = res.forecast(horizon.tolist())
        allforecasts = fore1.values.tolist()
        allforecasts = allforecasts[::-1]

        for j in np.arange(0,horizon):
            data1.iloc[j, [i]] = allforecasts[j]
     

data1.to_csv("src\state_space_python\mergedDataforAnalysis_statespace_COMPLETE.csv")   
data1.to_csv("data\mergedDataforAnalysis_statespace_COMPLETE.csv") 
