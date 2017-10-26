#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:11:16 2017

@author: sanshanxiashi
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
# 中位数
def mode_function(df):
    counts = mode(df)
    return counts[0][0]

#读取的数据，会自动生成int数据类型
feature_data = pd.read_csv('./fusai_data/feature_data_2017_3456_7.csv',dtype={'link_ID':str})


def mape(pred,real):
    # c=real.get_label()
    result=np.sum(np.abs(pred-real)/real)/len(real)
    # print('mape:',result)
    return result


def get_mape_func(df):
    max_v = df.max()
    min_v = df.min()
    print("min_v:", min_v, "max_v:", max_v)
    df_len = len(df)
    mape_result = 100
    result_values = 0
    for i in np.arange(min_v,max_v,0.1):
        print("   now_mape:", i)
        temp_list = np.array([i]*df_len)
        temp_result = mape(temp_list,df)
        if temp_result < mape_result:
            mape_result = temp_result
            result_values = i
    return result_values



######提取mape值
train3 = feature_data.loc[(feature_data.time_interval_month==3),:]
train3 = train3.loc[((feature_data.time_interval_begin_hour>=6)&(feature_data.time_interval_begin_hour<=8))
                    |((feature_data.time_interval_begin_hour>=13)&(feature_data.time_interval_begin_hour<=15))
                    |((feature_data.time_interval_begin_hour>=16)&(feature_data.time_interval_begin_hour<=18)),:]
train3_mape = train3.groupby(['link_ID','time_interval_begin_hour'])[
    'travel_time'].agg([('mape_v',get_mape_func)]).reset_index()

train3 = pd.merge(train3, train3_mape, on=['link_ID','time_interval_begin_hour'], how='left')
train3.to_csv('./lys/best_mape/mape_train3.csv',index=None)


train4 = feature_data.loc[(feature_data.time_interval_month==4),:]
train4 = train4.loc[((feature_data.time_interval_begin_hour>=6)&(feature_data.time_interval_begin_hour<=8))
                    |((feature_data.time_interval_begin_hour>=13)&(feature_data.time_interval_begin_hour<=15))
                    |((feature_data.time_interval_begin_hour>=16)&(feature_data.time_interval_begin_hour<=18)),:]
train4_mape = train4.groupby(['link_ID','time_interval_begin_hour'])[
    'travel_time'].agg([('mape_v',get_mape_func)]).reset_index()

train4 = pd.merge(train4, train4_mape, on=['link_ID','time_interval_begin_hour'], how='left')
train4.to_csv('./lys/best_mape/mape_train4.csv',index=None)




train5 = feature_data.loc[(feature_data.time_interval_month==5),:]
train5 = train5.loc[((feature_data.time_interval_begin_hour>=6)&(feature_data.time_interval_begin_hour<=8))
                    |((feature_data.time_interval_begin_hour>=13)&(feature_data.time_interval_begin_hour<=15))
                    |((feature_data.time_interval_begin_hour>=16)&(feature_data.time_interval_begin_hour<=18)),:]
train5_mape = train5.groupby(['link_ID','time_interval_begin_hour'])[
    'travel_time'].agg([('mape_v',get_mape_func)]).reset_index()

train5 = pd.merge(train5, train5_mape, on=['link_ID','time_interval_begin_hour'], how='left')
train5.to_csv('./lys/best_mape/mape_train5.csv',index=None)


train6 = feature_data.loc[(feature_data.time_interval_month==6),:]
train6 = train6.loc[((feature_data.time_interval_begin_hour>=6)&(feature_data.time_interval_begin_hour<=8))
                    |((feature_data.time_interval_begin_hour>=13)&(feature_data.time_interval_begin_hour<=15))
                    |((feature_data.time_interval_begin_hour>=16)&(feature_data.time_interval_begin_hour<=18)),:]
train6_mape = train6.groupby(['link_ID','time_interval_begin_hour'])[
    'travel_time'].agg([('mape_v',get_mape_func)]).reset_index()
    
train6 = pd.merge(train6, train6_mape, on=['link_ID','time_interval_begin_hour'], how='left')
train6.to_csv('./lys/best_mape/mape_train6.csv',index=None)



