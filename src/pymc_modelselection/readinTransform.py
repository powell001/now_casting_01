import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import random
from myhelpers import printme

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 400)


###############################
# Read in and Tranform data
# Make sure to get the right starting data and extend it if necessary
###############################

def collecttransform(diff_gdp_total = True):

    # use C:\Users\jpark\VSCode\now_casting_01\src\state_space_python\local_linear_trend_data_quarterly
    # to extend gdp data

    data1 = pd.read_csv("data\mergedDataforAnalysis_statespace_COMPLETE.csv", index_col=[0])

    numcols = data1.shape[1]

    # add monthly (mo) to monthly data
    month_columns = pd.read_csv("data\\a0_combinedMonthly.csv", index_col=[0])
    data1.columns = [f'{i}_monthly' if i in month_columns else f'{i}' for i in data1.columns]

    # remove season data
    data1.drop(columns=['gdp_total_season', 'imports_goods_services_season', 'household_cons_season', 'gov_consumption_season', 'investments_season', 'gpd_invest_business_households_season', 
                        'gov_invest_season', 'change_supply_season', 'exports_goods_services_season'], inplace=True)

    printme(data1)

    ###########################################
    # diff these
    ###########################################
    # differenced data
    diffdatacopy = data1.copy()
    diffthesecols = ['gdp_total']
    diff_data1 = diffdatacopy[diffthesecols]

    diff_data1 = diff_data1.diff()
    diff_data1.columns = ["diff_" + x for x in diff_data1.columns]
   

    # merge back
    data1 = pd.concat([data1, diff_data1], axis=1)


    # lag these (real values wont be available)
    lagthese = ['imports_goods_services', 'household_cons', 'gov_consumption', 'investments',
                'gpd_invest_business_households', 'gov_invest', 'change_supply', 'exports_goods_services']

    lag_data1 = data1.copy()
    data1[lagthese] = lag_data1[lagthese].shift(1)
    data1.columns = [f'lag_{i}' if i in lagthese else f'{i}' for i in data1.columns]

    # lagged gdp_total but keep unlagged of course
    data1['lag_gdp_total'] = data1['gdp_total'].shift(1)

    printme(data1)

    # correlations
    corr1 = data1.corr()
    corr1.to_csv('output_csvs_etc\correlations_all.csv')

    ##############################
    # create 5 random features
    ##############################
    rws = data1.shape[0]
    x = pd.DataFrame(random.randint(100, size=(5, rws))).T
    x.columns = ["random_" + str(x1)  for x1 in np.arange(0, 5)]
    x.index = data1.index
    data1 = data1.join(x)

    ### Add trend ###
    data1['trend'] = np.arange(0, data1.shape[0])

    ### Normalize (Including GDP) 
    normalized_data1 = (data1 - data1.mean())/data1.std()
    normalized_data1.to_csv("modelDATA.csv")

    # ##############################
    # # PYMC models
    # ##############################
    df1 = normalized_data1.copy()

    selectthese = normalized_data1.columns # select all columns
    df1 = df1[selectthese]
    printme(df1)

    ##############
    # add dummy
    ##############
    # extreems dummy
    df1['dummy_downturn'] = 0
    df1.loc['2009-01-01', 'dummy_downturn'] = 1
    df1.loc['2020-01-01', 'dummy_downturn'] = 1
    df1.loc['2020-04-01', 'dummy_downturn'] = 1
    # df1.loc['2020-07-01', 'dummy_downturn'] = 1
    # df1.loc['2021-04-01', 'dummy_downturn'] = 1
    # df1.loc['2021-07-01', 'dummy_downturn'] = 1

    df1.to_csv("output_csvs_etc\df1.csv")

    ##############
    # Examine model data
    ##############
    # few observations:

    # too_few_obs = ['BusinessOutlook_Industry_monthly', 'BusinessOutlook_Retail_monthly', 'IMP_advanceEconomies_monthly', 'EXP_advancedEconomies_monthly', 'IMP_EuroArea_monthly', 'Exp_EuroArea_monthly']
    # df1.drop(columns = too_few_obs, inplace = True)
    # df1.dropna(inplace=True)

    # highly correlated:
    high_corr = [
                    'lag_imports_goods_services',
                    'M3_1_monthly', 
                    'lag_investments',
                    'Koopbereidheid_3_monthly', 
                    'CPI_1_monthly',
                    'MaandmutatieCPI_3_monthly',
                    'M3_2_monthly',
                    'IMP_advanceEconomies_monthly', 
                    'EXP_advancedEconomies_monthly']

    df1.drop(columns = high_corr, inplace = True)
    # printme(df1)

    corr1 = df1.corr()
    corr1.to_csv('output_csvs_etc\correlations.csv')

    df1.to_csv("output_csvs_etc\premodel_data.csv")

    return df1


df1 = collecttransform()



df1.to_csv("jeff1.csv")
printme(df1)

###############
# reconstitutes1
################
# print(stdGDP, meanGDP)

# reconstitute1 = (df1['gdp_total'] * stdGDP) + meanGDP
# gdp2 = np.append(firstGDP, reconstitute1.dropna())
# print(np.cumsum(gdp2))

# print(firstGDP)