# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:58:47 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
import copy
import os

soure_path = os.getcwd()
data_path = os.path.dirname(soure_path)


def mape1(y,d):
    c=d
    #print("len(c)",len(c))
    result= np.sum(np.abs(y-c)/c)/len(c)
    return result
    


#feature_data_all = pd.read_csv('C:/Users/Administrator/Desktop/zl/ali/fusai_data/feature_data_2017_456_7.csv',dtype={'link_ID':str})
#
#feature_data7 =feature_data_all.loc[(feature_data_all.time_interval_month==7)&
#                                  ((feature_data_all.time_interval_begin_hour == 8)|
#                                  ((feature_data_all.time_interval_begin_hour == 15))|
#                                  ((feature_data_all.time_interval_begin_hour == 18))),:]
                                  
#for label_lengh in range(1,29,1):
#pred5_30_best = pd.read_csv(data_path+'/data/0.27139_m5/xgb_gbm_static_mape_lstm_ronghe_M6_ZhuLei.csv',dtype={'link_ID':str})
#
#
#pred5_xgb_1p = pd.read_csv(data_path+'/data/gbm_pred_ori_model_4train_52test_m6_first_point.csv',dtype={'link_ID':str})
'''
    5-6月
'''
pred5_30_best = pd.read_csv(data_path+'/result/'+'m7_final_into15.csv',dtype={'link_ID':str})

for label_lengh in range(2,0,-1):
    #label_lengh=1
    #test_pred5 = pd.read_csv('C:/Users/Administrator/Desktop/zl/ali/fusai/9_13/jiaocha'+'/LSTM_pred_m6_'+str(label_lengh)+'.csv',dtype={'link_ID':str})
    test_pred5 = pd.read_csv(data_path+'/data/instead_qian_2/'+'/LSTM_pred_m7_'+str(label_lengh)+'.csv',dtype={'link_ID':str})
    
    
    pred5_now = test_pred5.loc[test_pred5.time_interval_minutes<(label_lengh*2)]
    best = pred5_30_best.loc[pred5_30_best.time_interval_minutes<(label_lengh*2)]
         
#    print("前"+str(label_lengh)+"原始点 mape",  mape1(best['final_pred'], best['travel_time']))
#    print("前"+str(label_lengh)+"现在点 mape", mape1(pred5_now['lstm_pred'+str(label_lengh)], pred5_now['travel_time']) ) 
#    
    
    pred5_30_new = copy.deepcopy(pred5_30_best)
    pred5_30_new = pd.merge(pred5_30_new,test_pred5, on=['link_ID','time_interval_day', 'time_interval_begin_hour',
       'time_interval_minutes'], how='left')
    pred5_30_new = pred5_30_new.loc[:,
            ['link_ID', 'time_interval_day', 'time_interval_begin_hour',
       'time_interval_minutes', 
       'time_interval_begin', 'time_interval_month','travel_time_x', 'final_pred', 'lstm_pred'+str(label_lengh)]
    ]
    
    pred5_30_new.loc[pred5_30_new.time_interval_minutes<(label_lengh*2),'final_pred'] = pred5_30_new.loc[pred5_30_new.time_interval_minutes<(label_lengh*2),'lstm_pred'+str(label_lengh)].values
    pred5_30_new.rename(columns={ 'travel_time_x':'travel_time'}, inplace=True) 
    
    pred5_30_new = pred5_30_new.loc[:,
            ['link_ID', 'time_interval_day', 'time_interval_begin_hour',
       'time_interval_minutes', 
       'time_interval_begin', 'time_interval_month','travel_time', 'final_pred']
    ]
    
    mape_new = mape1(pred5_30_new['final_pred'], pred5_30_new['travel_time'])
    mape_old = mape1(pred5_30_best['final_pred'], pred5_30_best['travel_time'])
    print("     第",label_lengh,"轮的mape_new： ", mape_new)
    print("     最好的结果：", mape_old)
    
#    if mape_new < mape_old:
    pred5_30_best.loc[pred5_30_best.time_interval_minutes<(label_lengh*2),'final_pred'] = pred5_30_new.loc[pred5_30_new.time_interval_minutes<(label_lengh*2),'final_pred']
    print(label_lengh,"   更优秀！")
    print("final", mape1(pred5_30_best['final_pred'], pred5_30_best['travel_time']))
#    else:
#    print(label_lengh,"   不优秀..")
    print("")

#保存最优结果
print("!!", pred5_30_best.isnull().sum())
pred5_30_best.to_csv(data_path+'/result/'+'m7_final_instead_all.csv',index=False)