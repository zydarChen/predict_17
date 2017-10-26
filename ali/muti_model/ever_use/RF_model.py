# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 16:45:16 2017

@author: Administrator
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
#根据week的7个值 派生出一个dataframe(7列)， n*7
#week = pd.get_dummies(feature_data['time_interval_week'],prefix='week')

del feature_data['time_interval_week']
#把week的列添加到feature_data中
#feature_data = pd.concat([feature_data,week],axis=1)
#print(feature_data.head())

data_all = feature_data
#(7823975, 11)
#每个link，每月每天每小时的均值
data_mean = data_all.groupby(['link_ID','time_interval_month','time_interval_day',
                              'time_interval_begin_hour'])['travel_time'].agg([('mean_',np.mean)]).reset_index()

data_all= pd.merge(data_all, data_mean, on=['link_ID','time_interval_month','time_interval_day',
                              'time_interval_begin_hour'],how='left')
    
'''
去除大于均值1.5倍的记录
'''
data_noExp = data_all.loc[data_all.travel_time<= 1.135*data_all.mean_,:]

data_ori = feature_data
feature_data = data_noExp
feature_data.pop('mean_')
#(7079208, 10)



'''
正常跑模型
'''

#train
'''
训练集和其特征都要用 去除异常值后的数据
'''
train = pd.DataFrame()
#for curHour in range(1,24):
for curHour in [8,15,18]:
    print("train curHour", curHour)
    trainTmp = feature_data.loc[(feature_data.time_interval_month == 4)&
           (feature_data.time_interval_begin_hour==curHour),:]

    for i in [58,48,38,28,18,0]:
        tmp = feature_data.loc[(feature_data.time_interval_month == 4)&
                (feature_data.time_interval_begin_hour==curHour-1)
                                        &(feature_data.time_interval_minutes >= i),:]
        tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
                'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                    ('mode_%d' % (i), mode_function)]).reset_index()
        #train = pd.merge(train,tmp,on=['link_ID','time_interval_day','time_interval_begin_hour'],how='left')
        trainTmp = pd.merge(trainTmp,tmp,on=['link_ID','time_interval_day'],how='left')
        
    train = pd.concat([train,trainTmp], axis=0)
    print("train.shape", train.shape)

train_history = feature_data.loc[(feature_data.time_interval_month == 3),: ]
train_history = train_history.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                                ('mode_m', mode_function)]).reset_index()

train = pd.merge(train,train_history,on=['link_ID','time_interval_minutes'],how='left')
train_label = np.log1p(train.pop('travel_time'))
train_time = train.pop('time_interval_begin')





#test
'''
test的前一个月和前一个小时的特征 要去异常。 用于评测的test本身的数据要全
'''
test = pd.DataFrame()
for curHour in [8,15,18]:
    print("test curHour", curHour)
    testTmp = data_ori.loc[(data_ori.time_interval_month == 5)&
           (data_ori.time_interval_begin_hour==curHour),:]

    for i in [58,48,38,28,18,0]:
        tmp = feature_data.loc[(feature_data.time_interval_month == 5)&
                (feature_data.time_interval_begin_hour==curHour-1)
                                        &(feature_data.time_interval_minutes >= i),:]
        tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
                'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                    ('mode_%d' % (i), mode_function)]).reset_index()
        testTmp = pd.merge(testTmp,tmp,on=['link_ID','time_interval_day'],how='left')
    
    test = pd.concat([test,testTmp], axis=0)
    print("test.shape", test.shape)

test_history = feature_data.loc[(feature_data.time_interval_month == 4),: ]
test_history = test_history.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                                ('mode_m', mode_function)]).reset_index()

test = pd.merge(test,test_history,on=['link_ID','time_interval_minutes'],how='left')

test_label = np.log1p(test.pop('travel_time'))
test_time = test.pop('time_interval_begin')

train.drop(['time_interval_month'],inplace=True,axis=1)
test.drop(['time_interval_month'],inplace=True,axis=1)

train.isnull().sum()

#去掉link_ID
train.pop('link_ID')
test.pop('link_ID')

#填补缺省值
train = train.fillna(0)
test  = test.fillna(0)



#train
# 评价函数ln形式
def mape_ln1(y,d):
    #c=d.get_label()
    #print("y.len", len(y))
    c = d
    result= -np.sum(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)
    return "mape",result

def mape(y,d):
    c=d.get_label()
    result= -np.sum(np.abs(y-c)/c)/len(c)
    return "mape",result


from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


regr = RandomForestRegressor(random_state=9,n_jobs =10,
                        n_estimators = 40,
                        min_samples_leaf= 170,
                        max_depth = 11,
                        min_samples_split = 80,
                        max_features = 10
                        )
regr.fit(train.values, train_label.values)
predict = regr.predict(test.values)
            
mape_result = mape_ln1(predict , test_label)
print( mape_result)


'''''
原始数据跑：
     ('mape', -0.319280434726493)
     
剔除异常值：
    ('mape', -0.2732387686554558)

