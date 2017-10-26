# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 14:28:21 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def AddBaseTimeFeature(df):

    #添加column time_interval_begin, 删除data和time_interval
    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))
    df = df.drop(['date', 'time_interval'], axis=1)
    
    df['time_interval_year'] = df['time_interval_begin'].map(lambda x: x.strftime('%Y'))
    df['time_interval_month'] = df['time_interval_begin'].map(lambda x: x.strftime('%m'))
    df['time_interval_day'] = df['time_interval_begin'].map(lambda x: x.day)
    df['time_interval_begin_hour'] = df['time_interval_begin'].map(lambda x: x.strftime('%H'))
    df['time_interval_minutes'] = df['time_interval_begin'].map(lambda x: x.strftime('%M'))
    # Monday=1, Sunday=7
    df['time_interval_week'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    return df



link_info = pd.read_table('./gy_contest_link_info.txt',sep=';',dtype={'link_ID':str})
link_info = link_info.sort_values('link_ID')

#train_data 2017.456All 和 2016.7All
training_data = pd.read_table('./fusai_data/quaterfinal_gy_cmp_training_traveltime.txt',sep=';',dtype={'link_ID':str})
print(training_data.shape)
training_data = pd.merge(training_data,link_info,on='link_ID')

#test_data 2017.7part
testing_data = pd.read_table('./fusai_data/submit_seg1.txt',sep='#',header=0,dtype={'link_ID':str})
testing_data = pd.merge(testing_data,link_info,on='link_ID')
testing_data['travel_time'] = np.NaN
print(testing_data.shape)

#将train_data和test_data放在一起 
feature_date = pd.concat([training_data,testing_data],axis=0)
#将feature_date 根据link_ID和time_interval排序
feature_date = feature_date.sort_values(['link_ID','time_interval'])

#将feature_date中 'time_interval'时分秒分开
feature_data_date0 = AddBaseTimeFeature(feature_date) #(10810244, 11)

#区分2016和2017年数据
feature_data_date0_2017 = feature_data_date0.loc[feature_data_date0.time_interval_year=='2017',:]
#(8403724, 12)
feature_data_date0_2016 = feature_data_date0.loc[feature_data_date0.time_interval_year=='2016',:]
#(2406520, 12)

#调整数据类型 linkid->str, 时间->int
#2017
feature_data_date0_2017.to_csv('./fusai_data/feature_data_2017_456_7.csv',index=False)
feature_data_date0_2017_tmp = pd.read_csv('./fusai_data/feature_data_2017_456_7.csv',dtype={'link_ID':str, 'time_interval_year':int, 'time_interval_month':int, 'time_interval_begin_hour': int, 'time_interval_minutes': int})
feature_data_date0_2017_tmp.to_csv('./fusai_data/feature_data_2017_456_7.csv',index=False)
feature_data_date0_2017_456_7 = feature_data_date0_2017_tmp
#2016
feature_data_date0_2016.to_csv('./fusai_data/feature_data_2016_7all.csv',index=False)
feature_data_date0_2016_tmp = pd.read_csv('./fusai_data/feature_data_2016_7all.csv',dtype={'link_ID':str, 'time_interval_year':int, 'time_interval_month':int, 'time_interval_begin_hour': int, 'time_interval_minutes': int})
feature_data_date0_2016_tmp.to_csv('./fusai_data/feature_data_2016_7all.csv',index=False)

#把初赛3月份的数据加进来
feature_data_2017_345_6 = pd.read_csv('./fusai_data/feature_data_2017_345_6.csv',dtype={'link_ID':str})
feature_data_2017_3All = feature_data_2017_345_6.loc[feature_data_2017_345_6.time_interval_month==3,:]
feature_data_2017_3All['time_interval_year'] = 2017

feature_data_2017_3456_7 = pd.concat([feature_data_2017_3All, feature_data_date0_2017_456_7], axis=0)

#指定列的顺序
columns = [ 'link_ID', 'length', 'width', 'link_class', 'time_interval_begin', 'time_interval_year',
       'time_interval_month','time_interval_day', 'time_interval_begin_hour', 
       'time_interval_minutes', 'time_interval_week',
        'travel_time']
 feature_data_2017_3456_7.to_csv('./fusai_data/feature_data_2017_3456_7.csv', index=False, columns=columns)










