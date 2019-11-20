import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from utils_global import *

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# plot setting
SMALL_SIZE = 16
MEDIUM_SIZE = 19
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # text size
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

if mode == 'train':
    from train import res as data
if mode == 'test':
    from predict import res as data
# check the result dataframe (need to include the 4 columns: 'winprob', 'plaprob', 'winstake', 'plastake')
data.shape
data.head()
data.tail()
# data['rdate'].describe()  # 2007-9-9 to 2019-1-6


# =============================================================================
# Back testing functions
# this will run on the whole dataframe with predictions and bet ratios
# =============================================================================
# a function of computing RMSE of win prediction & place prediction
def RMSE(df, col1, col2):
    '''
    Input: 
        df: the dataframe
        col1: colunm name of the predicted probability
        col2: colunm name of the indicator
    Output:
        root_mse: RMSE  
    '''
    root_mse = np.sqrt( ((df[col1] - df[col2]) ** 2).mean() )
    return root_mse

# a function of computing the return per game by REAL odds
def RETpg(df, col1, col2):
    '''
    Input: 
        df: the dataframe
        col1: colunm name of the win bet ratio
        col2: colunm name of the place bet ratio
    Output:
        ret: a series of returns of games
    '''
    win_ret = df[col1] * (df['ind_win'] * df['win'] - 1)
    pla_ret = df[col2] * (df['ind_pla'] * df['place'] - 1)
    # sum considering NA
    ret = win_ret.add(pla_ret, fill_value=0)
    # return per game
    return ret.sum()

# a function of computing the return per game by 'FAIR' odds (remove the 18% margin)
# so the difference from RETpg is to amplify real odds
def RETpg_fair(df, col1, col2):
    '''
    Input: 
        df: the dataframe
        col1: colunm name of the win bet ratio
        col2: colunm name of the place bet ratio
    Output:
        ret: a series of returns of games
    '''
    win_ret = df[col1] * (df['ind_win'] * df['win']/0.82 - 1)
    pla_ret = df[col2] * (df['ind_pla'] * df['place']/0.82 - 1)
    # sum considering NA
    ret = win_ret.add(pla_ret, fill_value=0)
    # return per game
    return ret.sum()

# a function to do the back testing (RMSE of win&place and summary of returns)
#%% use the above functions RMSE, RETpg, RETpg_fair
def backtest(data, wpcol, ppcol, wstake, pstake):
    '''
    Input: 
        data: the dataframe
        wpcol: colunm name of the win probability
        ppcol: colunm name of the place probability
        wstake: colunm name of the win bet ratio
        pstake: colunm name of the place bet ratio
    Output:
        y: a summary series
    '''
    groups = data.groupby(['rdate', 'rid'])
    
    ### 1. average of RMSE of win prob over games
    rmse_win = groups.apply(RMSE, col1=wpcol, col2='ind_win')
    avgrmse_win = rmse_win.mean()
    
    ### 2. average RMSE of place prob over games
    rmse_pla = groups.apply(RMSE, col1=ppcol, col2='ind_pla')
    avgrmse_pla = rmse_pla.mean()
    
    ### 3. compute and summarize return by REAL odds
    retpg = groups.apply(RETpg, col1=wstake, col2=pstake)
    # cumulative wealth
    cum_wealth = np.nancumprod(1+retpg)    
    # final wealth
    finalwealth = cum_wealth[-1]
    # total profit
    totalprofit = finalwealth - 1
    # bet ratio per game
    ratiopg = groups[wstake].sum() + groups[pstake].sum()
    # bet amount per game
    costpg = ratiopg * (cum_wealth/(1+retpg)) 
    # mean return per dollar
    meanretpd = np.round(totalprofit/costpg.sum(), 4)
    
    ### 4. compute and summarize return by FAIR odds
    retpg_fair = groups.apply(RETpg_fair, col1=wstake, col2=pstake)
    # cumulative wealth
    cum_wealth_fair = np.nancumprod(1+retpg_fair)    
    # final wealth
    finalwealth_fair = cum_wealth_fair[-1]
    # total profit
    totalprofit_fair = finalwealth_fair - 1
    # mean return per dollar
    meanretpd_fair = np.round(totalprofit_fair/costpg.sum(), 4)

    # number of betting games
    ngames = (ratiopg!=0).sum()
    # number of betting horses
    nhorses = ((data['winstake']+data['plastake'])!=0).sum()

    # plot changes of cumulative wealth per game
    if ngames > 0:
        #plt.figure(figsize=(15,10))
        plt.figure(figsize=(9, 6))
        plt.plot(cum_wealth, color='blue', label='Real Odds')
        plt.plot(cum_wealth_fair, color='orange', label='Fair Odds')
        plt.grid(alpha=0.6)
        plt.legend(loc='upper right')
        plt.xlabel('Games')
        plt.ylabel('Cumulative Wealth')
        
    # create the summary series   
    y = pd.Series([np.round(avgrmse_win,4), np.round(avgrmse_pla,4), 
                   meanretpd, np.round(totalprofit,4), np.round(finalwealth,4), 
                   meanretpd_fair, np.round(totalprofit_fair,4), np.round(finalwealth_fair,4), 
                   nhorses, ngames],
                   index=['AverageRMSEwin','AverageRMSEpalce',
                          'MeanRetPerDollar(Real odds)','TotalProfit(Real odds)','FinalWealth(Real odds)',
                          'MeanRetPerDollar(Fair odds)','TotalProfit(Fair odds)','FinalWealth(Fair odds)',
                          'No.Horses','No.Games'])    
    print(y)
    
    return y

# run back testing
if __name__ == '__main__':
    # bt_result saves the summary of your prediction
    bt_result = backtest(data, 'win_prob', 'place_prob', 'winstake', 'plastake')
    bt_result.to_csv('backtesting_summary_Python.csv')



