import pandas as pd
import numpy as np

# transform monthly data to quarterly data
# should start at 4/1/1996

def creatXdata():

    data = pd.read_csv('src\\state_space_python\\a0_combinedMonthly_statespace.csv', index_col=[0])
    data = data.loc["1996-02-01":,:]
    cols = data.columns
    print(cols)

    combined = []
    for i in cols:
        print(i)
        data1 = data[i]
        print(data1)
        rows = data1.shape[0]
        data1 = data1.groupby(np.arange(rows)//3).mean()

        data1.columns = i
        data1.index = pd.date_range(start='04/01/1996', end='10/01/2024', freq="Q").to_period('Q')
        combined.append(data1)

    Xdata = pd.concat(combined, axis=1)
    Xdata.index = pd.date_range(start='04/01/1996', end='10/01/2024', freq="Q").to_period('Q')
    Xdata.index = pd.PeriodIndex(Xdata.index, freq='Q').to_timestamp()

    Xdata.to_csv("src\\state_space_python\\Xdata.csv")

    return Xdata

Xdata = creatXdata()

#################################
# Combine with Quarterly data
#################################

Qtdata = pd.read_csv('src\\state_space_python\\a0_combinedQuarterly_statespace.csv', index_col=[0])

Qtdata.index = pd.date_range(start='04/01/1995', end='07/01/2024', freq="Q").to_period('Q')
Qtdata.index = pd.PeriodIndex(Qtdata.index, freq='Q').to_timestamp()

Qtdata = Qtdata.loc["04/01/1995":, :]

data2 = pd.merge(Qtdata, Xdata, left_index = True, right_index = True, how = 'outer')
data2.to_csv("src\\state_space_python\\mergedDataforAnalysis_statespace.csv")
data2.to_csv("data\\mergedDataforAnalysis_statespace.csv")
