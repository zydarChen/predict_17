# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:57:50 2017

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
feature_data = pd.read_csv('E:/lys/Gui_Zhou/data/all_data_M34567.csv',dtype={'link_ID':str})
#根据week的7个值 派生出一个dataframe(7列)， n*7
#week = pd.get_dummies(feature_data['time_interval_week'],prefix='week')
#
#del feature_data['time_interval_week']
##把week的列添加到feature_data中
#print("week onehot")
#feature_data = pd.concat([feature_data,week],axis=1)
#print(feature_data.head())



'''
train data 4月训练
'''

train = pd.DataFrame()
#6月训练集
train4 = pd.DataFrame()
for curHour in [8,15,18]:
    print("train4 curHour", curHour)
    trainTmp = feature_data.loc[(feature_data.time_interval_month == 4)&
           (feature_data.time_interval_begin_hour==curHour)
          # &(feature_data.time_interval_day<=15)
           ,:]

    for i in [58,48,38,28,18,0]:
        tmp = feature_data.loc[(feature_data.time_interval_month == 4)&
                (feature_data.time_interval_begin_hour==curHour-1)
                                        &(feature_data.time_interval_minutes >= i),:]
        tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
                'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                    ('mode_%d' % (i), mode_function)]).reset_index()
        #train = pd.merge(train,tmp,on=['link_ID','time_interval_day','time_interval_begin_hour'],how='left')
        trainTmp = pd.merge(trainTmp,tmp,on=['link_ID','time_interval_day'],how='left')
        
    train4 = pd.concat([train4,trainTmp], axis=0)
    print("     train4.shape", train4.shape)

train4_history = feature_data.loc[(feature_data.time_interval_month == 3),: ]
train4_history = train4_history.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                                ('mode_m', mode_function)]).reset_index()

train4 = pd.merge(train4,train4_history,on=['link_ID','time_interval_minutes'],how='left')

train_history2 = feature_data.loc[(feature_data.time_interval_month == 3),: ]
train_history2 = train_history2.groupby(['link_ID', 'time_interval_begin_hour'])[
            'travel_time'].agg([ ('median_h', np.median),
                                ('mode_h', mode_function)]).reset_index()
            
train4 = pd.merge(train4, train_history2,on=['link_ID','time_interval_begin_hour'],how='left')
print("train4.shape", train4.shape)
train = train4

train_label = np.log1p(train.pop('travel_time'))
train_time = train.pop('time_interval_begin')

train.drop(['time_interval_month'],inplace=True,axis=1)
train_link=train.pop('link_ID') #(253001, 35)
print("train.shape", train.shape)




'''
test   评测6月整月    [402]   validation_0-mape:-0.283801
'''
test = pd.DataFrame()
for curHour in [8,15,18]:
    print("test curHour", curHour)
    testTmp = feature_data.loc[(feature_data.time_interval_month == 6)&
           (feature_data.time_interval_begin_hour==curHour)
           ,:]

    for i in [58,48,38,28,18,0]:
        tmp = feature_data.loc[(feature_data.time_interval_month == 6)&
                (feature_data.time_interval_begin_hour==curHour-1)
                                        &(feature_data.time_interval_minutes >= i),:]
        tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
                'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                    ('mode_%d' % (i), mode_function)]).reset_index()
        testTmp = pd.merge(testTmp,tmp,on=['link_ID','time_interval_day'],how='left')
    
    test = pd.concat([test,testTmp], axis=0)
    print("test.shape", test.shape)

test_history = feature_data.loc[(feature_data.time_interval_month == 5),: ]
test_history = test_history.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                                ('mode_m', mode_function)]).reset_index()

test = pd.merge(test,test_history,on=['link_ID','time_interval_minutes'],how='left')

test_history2 = feature_data.loc[(feature_data.time_interval_month == 5),: ]
test_history2 = test_history2.groupby(['link_ID', 'time_interval_begin_hour'])[
            'travel_time'].agg([ ('median_h', np.median),
                                ('mode_h', mode_function)]).reset_index()
            
test = pd.merge(test,test_history2,on=['link_ID','time_interval_begin_hour'],how='left')

test_label = np.log1p(test.pop('travel_time'))
test_time = test.pop('time_interval_begin')


test.drop(['time_interval_month'],inplace=True,axis=1)

#去掉link_ID
test_link=test.pop('link_ID')


def mape_ln(y,d):
    c=d.get_label()
    #c=d
    result= -np.sum(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)
    return "mape",result

def mape(y,d):
    c=d.get_label()
    result= -np.sum(np.abs(y-c)/c)/len(c)
    return "mape",result

def mape1(y,d):
    c=d
    result= -np.sum(np.abs(y-c)/c)/len(c)
    return "mape",result

def mape_object(y,d):

    g=1.0*np.sign(y-d)/d
    h=1.0/d
    return -g,h


def mape_ln1(y,d):
    #c=d.get_label()
    c=d
    result= -np.sum(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)
    return "mape",result


import xgboost as xgb
xlf = xgb.XGBRegressor(max_depth=8,
                       learning_rate=0.01,
                       n_estimators=1000,
                       silent=True,
                       objective=mape_object,
                       #objective='reg:linear',
                       nthread=-1,
                       gamma=0,
                       min_child_weight=6,
                       max_delta_step=0,
                       subsample=0.9,
                       colsample_bytree=0.8,
                       colsample_bylevel=1,
                       reg_alpha=1e0,
                       reg_lambda=0,
                       scale_pos_weight=1,
                       seed=9,
                       missing=None)

xlf.fit(train.values, train_label.values, eval_metric=mape_ln, 
        verbose=True, eval_set=[(test.values, test_label.values)],
        early_stopping_rounds=10)


'''
预测sub，并保存结果   5月下
'''
sub = pd.DataFrame()
for curHour in [8,15,18]:
    print("sub curHour", curHour)
    subTmp = feature_data.loc[(feature_data.time_interval_month == 5)&
           (feature_data.time_interval_begin_hour==curHour)
           #&(feature_data.time_interval_day>15)
           ,:]

    for i in [58,48,38,28,18,0]:
        tmp = feature_data.loc[(feature_data.time_interval_month == 5)&
                (feature_data.time_interval_begin_hour==curHour-1)
                                        &(feature_data.time_interval_minutes >= i),:]
        tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
                'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                    ('mode_%d' % (i), mode_function)]).reset_index()
        subTmp = pd.merge(subTmp,tmp,on=['link_ID','time_interval_day'],how='left')
    
    sub = pd.concat([sub,subTmp], axis=0)
    print("sub.shape", sub.shape)

sub_history = feature_data.loc[(feature_data.time_interval_month == 4),: ]
sub_history = sub_history.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                                ('mode_m', mode_function)]).reset_index()

sub = pd.merge(sub,sub_history,on=['link_ID','time_interval_minutes'],how='left')

sub_history2 = feature_data.loc[(feature_data.time_interval_month == 4),: ]
sub_history2 = sub_history2.groupby(['link_ID', 'time_interval_begin_hour'])[
            'travel_time'].agg([('median_h', np.median),
                                ('mode_h', mode_function)]).reset_index()
            
sub = pd.merge(sub,sub_history2,on=['link_ID','time_interval_begin_hour'],how='left')
sub_label = np.log1p(sub.pop('travel_time'))
sub_time = sub.pop('time_interval_begin')

sub.drop(['time_interval_month'],inplace=True,axis=1)
#去掉link_ID
sub_link = sub.pop('link_ID')

#预测
sub_pred = xlf.predict(sub.values, ntree_limit=xlf.best_iteration)
mape_ln1(sub_pred, sub_label) #('mape', -0.27325180044232567)

sub_out = pd.concat([sub_link, sub], axis=1)
sub_out = pd.concat([sub_out,np.expm1(sub_label)],axis=1)
sub_out['xgb_pred'] = np.expm1(sub_pred)
sub_out.to_csv('./predict_result/xgb_pred_m5.csv', index=False)



'''
预测sub，并保存结果   6月整月
'''
sub = pd.DataFrame()
for curHour in [8,15,18]:
    print("sub curHour", curHour)
    subTmp = feature_data.loc[(feature_data.time_interval_month == 6)&
           (feature_data.time_interval_begin_hour==curHour)
           ,:]

    for i in [58,48,38,28,18,0]:
        tmp = feature_data.loc[(feature_data.time_interval_month == 6)&
                (feature_data.time_interval_begin_hour==curHour-1)
                                        &(feature_data.time_interval_minutes >= i),:]
        tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
                'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                    ('mode_%d' % (i), mode_function)]).reset_index()
        subTmp = pd.merge(subTmp,tmp,on=['link_ID','time_interval_day'],how='left')
    
    sub = pd.concat([sub,subTmp], axis=0)
    print("sub.shape", sub.shape)

sub_history = feature_data.loc[(feature_data.time_interval_month == 5),: ]
sub_history = sub_history.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                                ('mode_m', mode_function)]).reset_index()

sub = pd.merge(sub,sub_history,on=['link_ID','time_interval_minutes'],how='left')

sub_history2 = feature_data.loc[(feature_data.time_interval_month == 5),: ]
sub_history2 = sub_history2.groupby(['link_ID', 'time_interval_begin_hour'])[
            'travel_time'].agg([('median_h', np.median),
                                ('mode_h', mode_function)]).reset_index()
            
sub = pd.merge(sub,sub_history2,on=['link_ID','time_interval_begin_hour'],how='left')
sub_label = np.log1p(sub.pop('travel_time'))
sub_time = sub.pop('time_interval_begin')

sub.drop(['time_interval_month'],inplace=True,axis=1)
#去掉link_ID
sub_link = sub.pop('link_ID')

#预测
sub_pred = xlf.predict(sub.values, ntree_limit=xlf.best_iteration)
mape_ln1(sub_pred, sub_label)

sub_out = pd.concat([sub_link, sub], axis=1)
sub_out = pd.concat([sub_out,np.expm1(sub_label)],axis=1)
sub_out['xgb_pred'] = np.expm1(sub_pred)
sub_out.to_csv('./predict_result/xgb_pred_m6.csv', index=False)


'''
预测sub，并保存结果   7月上
'''
sub = pd.DataFrame()
for curHour in [8,15,18]:
    print("sub curHour", curHour)
    subTmp = feature_data.loc[(feature_data.time_interval_month == 7)&
           (feature_data.time_interval_begin_hour==curHour)
          # &(feature_data.time_interval_day<=15)
           ,:]

    for i in [58,48,38,28,18,0]:
        tmp = feature_data.loc[(feature_data.time_interval_month == 7)&
                (feature_data.time_interval_begin_hour==curHour-1)
                                        &(feature_data.time_interval_minutes >= i),:]
        tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
                'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
                                    ('mode_%d' % (i), mode_function)]).reset_index()
        subTmp = pd.merge(subTmp,tmp,on=['link_ID','time_interval_day'],how='left')
    
    sub = pd.concat([sub,subTmp], axis=0)
    print("sub.shape", sub.shape)

sub_history = feature_data.loc[(feature_data.time_interval_month == 5),: ]
sub_history = sub_history.groupby(['link_ID', 'time_interval_minutes'])[
            'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                                ('mode_m', mode_function)]).reset_index()

sub = pd.merge(sub,sub_history,on=['link_ID','time_interval_minutes'],how='left')

sub_history2 = feature_data.loc[(feature_data.time_interval_month == 5),: ]
sub_history2 = sub_history2.groupby(['link_ID', 'time_interval_begin_hour'])[
            'travel_time'].agg([('median_h', np.median),
                                ('mode_h', mode_function)]).reset_index()
            
sub = pd.merge(sub,sub_history2,on=['link_ID','time_interval_begin_hour'],how='left')
sub_label = np.log1p(sub.pop('travel_time'))
sub_time = sub.pop('time_interval_begin')

sub.drop(['time_interval_month'],inplace=True,axis=1)
#去掉link_ID
sub_link = sub.pop('link_ID')

#预测
sub_pred = xlf.predict(sub.values, ntree_limit=xlf.best_iteration)
mape_ln1(sub_pred, sub_label)

sub_out = pd.concat([sub_link, sub], axis=1)
sub_out = pd.concat([sub_out,np.expm1(sub_label)],axis=1)
sub_out['xgb_pred'] = np.expm1(sub_pred)
sub_out.to_csv('./predict_result/xgb_pred_m7.csv', index=False)






