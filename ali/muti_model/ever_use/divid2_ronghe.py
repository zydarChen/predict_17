# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:50:23 2017

@author: Administrator
"""

import time
import warnings
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from numpy import newaxis
import os
from scipy.stats import mode


# s_time = time.time()
soure_path = os.getcwd()
data_path = os.path.dirname(soure_path)
warnings.filterwarnings("ignore")


#数据处理
gbm_data5 = pd.read_csv(data_path+'/data/gbm_pred_m5.csv',dtype = {"link_ID":str})
gbm_data5['time_interval_month'] = 5
gbm_data6 = pd.read_csv(data_path+'/data/gbm_pred_m6.csv',dtype = {"link_ID":str})
gbm_data6['time_interval_month'] = 6
gbm_data = pd.concat([gbm_data5, gbm_data6], axis=0)
#gbm_data = gbm_data6
gbm_data = gbm_data.loc[:,['link_ID','time_interval_month','time_interval_day',
'time_interval_begin_hour','time_interval_minutes','travel_time', 'gbm_pred']]
gbm_data = gbm_data.sort_values(['link_ID','time_interval_month','time_interval_day',
'time_interval_begin_hour','time_interval_minutes']).reset_index()
 
xgb_data5 = pd.read_csv(data_path+'/data/xgb_pred_m5.csv',dtype = {"link_ID":str})
xgb_data5['time_interval_month'] = 5
xgb_data6 = pd.read_csv(data_path+'/data/xgb_pred_m6.csv',dtype = {"link_ID":str})
xgb_data6['time_interval_month'] = 6
xgb_data = pd.concat([xgb_data5, xgb_data6], axis=0)
#xgb_data = xgb_data6
xgb_data = xgb_data.sort_values(['link_ID','time_interval_month','time_interval_day',
'time_interval_begin_hour','time_interval_minutes']).reset_index()
xgb_data = xgb_data.loc[:,['xgb_pred']]    

 
lstm_data5 = pd.read_csv(data_path+'/data/LSTM_pred_m5.csv',dtype = {"link_ID":str})
lstm_data5['time_interval_month'] = 5     
lstm_data6 = pd.read_csv(data_path+'/data//LSTM_pred_m6.csv',dtype = {"link_ID":str}) 
lstm_data6['time_interval_month'] = 6  
lstm_data = pd.concat([lstm_data5, lstm_data6], axis=0)
#lstm_data = lstm_data6
lstm_data = lstm_data.loc[:,['link_ID','time_interval_month','time_interval_day',
'time_interval_begin_hour','time_interval_minutes','travel_time', 'lstm_pred']] 
lstm_data = lstm_data.sort_values(['link_ID','time_interval_month','time_interval_day',
'time_interval_begin_hour','time_interval_minutes']).reset_index()
 
 
static_data5 = pd.read_csv(data_path+'/data/new_sm/static_1_2_para_m5.csv',dtype = {"link_ID":str})
static_data5['time_interval_month'] = 5    
static_data6 = pd.read_csv(data_path+'/data/new_sm/static_1_2_m6.csv',dtype = {"link_ID":str})  
static_data6['time_interval_month'] = 6    
static_data = pd.concat([static_data5, static_data6], axis=0)
#static_data = static_data6
static_data = static_data.sort_values(['link_ID','time_interval_month','time_interval_day',
'time_interval_begin_hour','time_interval_minutes']).reset_index()
static_data = static_data.loc[:,[ 'static_1_2_pred']] 


mape_data5 = pd.read_csv(data_path+'/data/new_sm/mape_1_2_3_para_m5.csv',dtype = {"link_ID":str})
mape_data5['time_interval_month'] = 5    
mape_data6 = pd.read_csv(data_path+'/data/new_sm/mape_1_2_3_m6.csv',dtype = {"link_ID":str})  
mape_data6['time_interval_month'] = 6
mape_data = pd.concat([mape_data5, mape_data6], axis=0)
#mape_data = mape_data6
mape_data = mape_data.sort_values(['link_ID','time_interval_month','time_interval_day',
'time_interval_begin_hour','time_interval_minutes']).reset_index()
mape_data = mape_data.loc[:,[ 'mape_1_2_3_pred']] 


 
all_data = pd.concat([gbm_data, xgb_data , static_data, mape_data], axis=1)


all_data6 = all_data.loc[(all_data.time_interval_month==6),:]
all_data6 = pd.merge(all_data6, lstm_data6, on=['link_ID','time_interval_month','time_interval_day',
'time_interval_begin_hour','time_interval_minutes'],how='left').reset_index()

all_data6.drop('travel_time_y',axis=1,inplace=True)
all_data6.rename(columns={'travel_time_x':'travel_time', 
                          'static_1_2_pred': 'static',
                          'mape_1_2_3_pred': 'mape',
                          'gbm_pred':'gbm',
                          'xgb_pred':'xgb',
                          'lstm_pred':'lstm'}, inplace=True) 


def mape(true,pred):
    result = np.sum(np.abs(pred-true)/true)/len(true)
    return result


'''
xgb+ gbm
'''
print("     xgb: ", mape(all_data6['travel_time'], all_data6['xgb']))
print("     gbm: ", mape(all_data6['travel_time'], all_data6['gbm']))
all_data6['xgbm'] = all_data6['xgb']*0.5 + all_data6['gbm']*0.5
mape_xgbm = mape(all_data6['travel_time'], all_data6['xgbm'])
print("         xgbm: ", mape_xgbm)
print("         提升了", mape(all_data6['travel_time'],all_data6['gbm']) - mape_xgbm)    



'''
xgbm+ static
'''
print("     xgbm: ", mape(all_data6['travel_time'], all_data6['xgbm']))
print("     static: ", mape(all_data6['travel_time'], all_data6['static']))
all_data6['xgbms'] = all_data6['xgbm']*0.5 + all_data6['static']*0.5
mape_xgbms = mape(all_data6['travel_time'], all_data6['xgbms'])
print("         xgbms: ", mape_xgbms)
print("         提升了: ", mape_xgbm - mape_xgbms)    


'''
xgbms+ mape
'''
print("     xgbms: ", mape(all_data6['travel_time'], all_data6['xgbms']))
print("     mape: ", mape(all_data6['travel_time'], all_data6['mape']))
all_data6['xgbmsm'] = all_data6['xgbms']*0.5 + all_data6['mape']*0.5
mape_xgbmsm = mape(all_data6['travel_time'], all_data6['xgbmsm'])
print("         xgbmsm: ", mape_xgbmsm)
print("         提升了: ", mape_xgbms - mape_xgbmsm)    


'''
xgbmsm+ lstm
'''
print("     xgbmsm: ", mape(all_data6['travel_time'], all_data6['xgbmsm']))
print("     lstm: ", mape(all_data6['travel_time'], all_data6['lstm']))
all_data6['xgbmsml'] = all_data6['xgbmsm']*0.5 + all_data6['lstm']*0.5
mape_xgbmsml = mape(all_data6['travel_time'], all_data6['xgbmsml'])
print("         xgbmsml: ", mape_xgbmsml)
print("         提升了: ", mape_xgbmsm - mape_xgbmsml)  

all_data6 = all_data6.loc[:,['link_ID', 'time_interval_month',
       'time_interval_day', 'time_interval_begin_hour',
       'time_interval_minutes', 'travel_time', 'gbm', 'xgb', 'static', 'mape',
       'time_interval_begin', 'lstm', 'xgbm', 'xgbms', 'xgbmsm', 'xgbmsml']]
all_data6['final_pred'] = all_data6['xgbmsml']

all_data6.to_csv(data_path+'/result/ronghe_m6.csv', index=False)





'''
     xgb:  0.2837829717304182
     gbm:  0.2843165071282243
         xgbm:  0.2822124630763883
         提升了 0.002104044051836007
     xgbm:  0.2822124630763883
     static:  0.28333102091467327
         xgbms:  0.2773145542178065
         提升了:  0.004897908858581823
     xgbms:  0.2773145542178065
     mape:  0.28277912686226797
         xgbmsm:  0.2769120696435024
         提升了:  0.000402484574304085
     xgbmsm:  0.2769120696435024
     lstm:  0.2767654917356837
         xgbmsml:  0.27385487214113563
         提升了:  0.003057197502366771
'''


#ronge_bylstm = pd.read_csv(data_path+'/result/fusion_bylstm_res6.csv',dtype = {"link_ID":str})
#ronge_bylstm = pd.read_csv(data_path+'/result/ronghe_m6.csv',dtype = {"link_ID":str})
#
#
















    


    
    
    
    
    
    
    
    
    