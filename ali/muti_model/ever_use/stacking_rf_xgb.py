# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 20:06:56 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.ensemble import RandomForestRegressor  
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import KFold 
import lightgbm as lgb
import xgboost as xgb




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
# 中位数
def mode_function(df):
    counts = mode(df)
    return counts[0][0]

#读取的数据，会自动生成int数据类型
feature_data = pd.read_csv('../fusai_data/feature_data_2017_3456_7.csv',dtype={'link_ID':str})
#根据week的7个值 派生出一个dataframe(7列)， n*7
week = pd.get_dummies(feature_data['time_interval_week'],prefix='week')

del feature_data['time_interval_week']
#把week的列添加到feature_data中
print("week onehot")
feature_data = pd.concat([feature_data,week],axis=1)
#print(feature_data.head())




'''
    train data
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

train_history2 = feature_data.loc[(feature_data.time_interval_month == 3),: ]
train_history2 = train_history2.groupby(['link_ID', 'time_interval_begin_hour'])[
            'travel_time'].agg([ ('median_h', np.median),
                                ('mode_h', mode_function),('std_h', np.std)]).reset_index()
            
train = pd.merge(train,train_history2,on=['link_ID','time_interval_begin_hour'],how='left')

train_label = np.log1p(train.pop('travel_time'))
train_time = train.pop('time_interval_begin')

train.drop(['time_interval_month'],inplace=True,axis=1)
train_link=train.pop('link_ID')


'''
test   
'''

test = pd.DataFrame()
for curHour in [8,15,18]:
    print("test curHour", curHour)
    testTmp = feature_data.loc[(feature_data.time_interval_month == 5)&
           (feature_data.time_interval_begin_hour==curHour),:]

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

test_history2 = feature_data.loc[(feature_data.time_interval_month == 4),: ]
test_history2 = test_history2.groupby(['link_ID', 'time_interval_begin_hour'])[
            'travel_time'].agg([ ('median_h', np.median),
                                ('mode_h', mode_function),('std_h', np.std)]).reset_index()
            
test = pd.merge(test,test_history2,on=['link_ID','time_interval_begin_hour'],how='left')

test_label = np.log1p(test.pop('travel_time'))
test_time = test.pop('time_interval_begin')


test.drop(['time_interval_month'],inplace=True,axis=1)

#去掉link_ID
test_link=test.pop('link_ID')






def mape_object_lgbm(d,y):
    # print(d)
    # print(y)
    grad=1.0*(y-d)/d
    hess=1.0/d
    return grad,hess


def mape_object(y,d):

    g=1.0*np.sign(y-d)/d
    h=1.0/d
    return -g,h

# 评价函数ln形式
def mape_ln(y,d):
    c=d.get_label()
    #c=d
    result= -np.sum(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)
    return "mape",result

def mape_ln1(y,d):
    #c=d.get_label()
    c=d
    result= -np.sum(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)
    return "mape",result    
    
    
    
    #一个模型获得 一个列特征(预测结果)
def get_oof(clf, X_train, y_train, X_test):
    k = 5    
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]
    kf = KFold(n_splits=k, random_state=2017)
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((k,ntest))
        
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        print("i: ",i)
        kf_X_train = X_train[train_index]
        kf_y_train = y_train[train_index]
        kf_X_test = X_train[test_index]
            
        clf.fit(kf_X_train, kf_y_train)
            
        oof_train[test_index] = clf.predict(kf_X_test)
        oof_test_skf[i, :] = clf.predict(X_test)
            
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
            


#Stacking  
def stackModel(train, train_label, test, test_label):    
    train_y = train_label.values  # 训练标签  
    
    train_X = train.values  
    test_X = test.values  
    
    clfs = [
#            RandomForestRegressor(random_state=9,n_jobs =10,
#                        n_estimators = 40,
#                        min_samples_leaf= 170,
#                        max_depth = 11,
#                        min_samples_split = 80,
#                        max_features = 10
#                        ),  
            xgb.XGBRegressor(max_depth=8,
                       learning_rate=0.01,
                       n_estimators=437,
                       silent=True,
                       objective=mape_object,
                       #objective='reg:linear',
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
                       missing=None),  
            lgb.LGBMRegressor(num_leaves=32,
                             # max_depth=9,
                             max_bin=511,
                       learning_rate=0.01,
                       n_estimators=1013,
                       silent=True,
                       objective=mape_object_lgbm,
                       min_child_weight=6,
                       colsample_bytree=0.8,
                       reg_alpha=1e0,
                       reg_lambda=0)] 
            
            
    #训练过程  
    dataset_stack_train = np.zeros((train_X.shape[0],len(clfs))) # (253001, 3)
    dataset_stack_test = np.zeros((test_X.shape[0],len(clfs))) #(354134, 3)
    
    for j,clf in enumerate(clfs):
        print("j: ",j, " clfs: ",clfs[j])
        y_train, y_submission = get_oof(clf, train_X, train_y, test_X)
        train_tmp =y_train.reshape(len(y_train))  #(253001,1) ->  (253001,)
        dataset_stack_train[:,j] = train_tmp
        submission_tmp = y_submission.reshape(len(y_submission))
        dataset_stack_test[:,j] = submission_tmp 
        
    print("开始Stacking....")  
    #clf = RandomForestRegressor(n_estimators=1000,max_depth=8)  
    clf = xgb.XGBRegressor()
    clf.fit(dataset_stack_train, train_y)
    
    y_submission = clf.predict(dataset_stack_test)  
    print( mape_ln1(y_submission, test_label.values) )
    predictions = np.expm1(y_submission)  
    return predictions
    

test_pred = stackModel(train, train_label, test, test_label)

#sub_pred = stackModel(train, train_label, sub, sub_label) 


#剔除异常值：
#Out[131]: ('mape', -0.2735122220679948)
    
#   xgb.XGBRegressor(learning_rate=0.1, n_estimators=85, max_depth=3)
#    ('mape', -0.27012328388980655)
          
            
            
#原始值
#最终：('mape', -0.30711185693261117)
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    