# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 03:57:19 2017

@author: Administrator
"""

from __future__ import print_function

import time
import warnings
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import os
from scipy.stats import mode
import keras.callbacks as kcallbacks

# s_time = time.time()
soure_path = os.getcwd()
data_path = os.path.dirname(soure_path)
warnings.filterwarnings("ignore")

def mape(true,pred):
    result = np.sum(np.abs(pred-true)/true)/len(true)
    return result

def mode_function(df):
    counts = mode(df)
    return counts[0][0]

def quantile1(df):
    result = df.quantile(0.1)
    return result
def quantile2(df):
    result = df.quantile(0.2)
    return result
def quantile3(df):
    result = df.quantile(0.3)
    return result
def quantile4(df):
    result = df.quantile(0.4)
    return result
def quantile5(df):
    result = df.quantile(0.5)
    return result
def quantile6(df):
    result = df.quantile(0.6)
    return result
def quantile7(df):
    result = df.quantile(0.7)
    return result
def quantile8(df):
    result = df.quantile(0.8)
    return result
def quantile9(df):
    result = df.quantile(0.9)
    return result



def skew(df):

    result = df.skew()
    return result

def mad(df):
    result = df.mad()
    return result

def kurt(df):
    result = df.kurt()
    return result


def test():
    ori_feature_data = pd.read_csv(data_path + '/data' + '/quaterfinal_all_link_ori_data_0-24_replace_missval.csv',
                                   dtype={"link_ID": str})
    feature_data = pd.read_csv(data_path + '/data' + '/quaterfinal_all_link_ori_data_0-24_replace_missval.csv',
                                   dtype={"link_ID": str})
    ori_test_data = ori_feature_data.loc[(ori_feature_data.time_interval_month == 5) &
                                     (ori_feature_data.time_interval_day >= 15) &
                                     ((ori_feature_data.time_interval_begin_hour == 6) |
                                      (ori_feature_data.time_interval_begin_hour == 7) |
                                      (ori_feature_data.time_interval_begin_hour == 8) |
                                      (ori_feature_data.time_interval_begin_hour == 13) |
                                      (ori_feature_data.time_interval_begin_hour == 14) |
                                      (ori_feature_data.time_interval_begin_hour == 15) |
                                      (ori_feature_data.time_interval_begin_hour == 16) |
                                      (ori_feature_data.time_interval_begin_hour == 17) |
                                      (ori_feature_data.time_interval_begin_hour == 18)), :]
    test_data = feature_data.loc[(feature_data.time_interval_month == 5) &
                                     (feature_data.time_interval_day >= 15) &
                                     ((feature_data.time_interval_begin_hour == 6) |
                                      (feature_data.time_interval_begin_hour == 7) |
                                      (feature_data.time_interval_begin_hour == 8) |
                                      (feature_data.time_interval_begin_hour == 13) |
                                      (feature_data.time_interval_begin_hour == 14) |
                                      (feature_data.time_interval_begin_hour == 15) |
                                      (feature_data.time_interval_begin_hour == 16) |
                                      (feature_data.time_interval_begin_hour == 17) |
                                      (feature_data.time_interval_begin_hour == 18)), :]
    all = pd.merge(ori_test_data,test_data,on=['link_ID','time_interval_day',
                                               'time_interval_begin_hour','time_interval_minutes'],how='left')
    print(all[all.isnull().T.any().T])
def get_mape_func(df):
    # df = np.array(df)
    # df = np.log1p(df)
    max_v = df.max()
    min_v = df.min()
    df_len = len(df)
    if df_len == 1:
        return max_v
    mape_result = 100
    result_values = 0
    for i in np.arange(min_v,max_v,1.0):
        temp_list = np.array([i]*df_len)
        temp_result = mape(temp_list,df)
        if temp_result < mape_result:
            mape_result = temp_result
            result_values = i
    return result_values

def load_data(seq_len,pred_len):
    ori_feature_data = pd.read_csv(data_path+'/data'+'/all_data_M34567.csv',dtype = {"link_ID":str})
    #feature_data = pd.read_csv(data_path+'/data'+'/feature_data_2017_4567_replaceForHigherPoint.csv',dtype = {"link_ID":str})
    # feature_data = pd.read_csv(data_path+'/data'+'/all_data_M34567_replaceForHigherPoint.csv',dtype = {"link_ID":str})
    feature_data = ori_feature_data
    #load train_data
    train_data = feature_data.loc[(feature_data.time_interval_month == 4)&
                                  (feature_data.time_interval_day < 31) &
                                 (
                                  (feature_data.time_interval_begin_hour == 7)|
                                  (feature_data.time_interval_begin_hour == 8)|
                                  (feature_data.time_interval_begin_hour == 14)|
                                  (feature_data.time_interval_begin_hour == 15)|
                                  (feature_data.time_interval_begin_hour == 17)|
                                  (feature_data.time_interval_begin_hour == 18)),:]
    # train_data = feature_data.loc[(feature_data.time_interval_month == 4) &
    #                               (feature_data.time_interval_day < 31) &
    #                               (
    #                                   (feature_data.time_interval_begin_hour == 7) |
    #                                   (feature_data.time_interval_begin_hour == 14)|
    #                                   (feature_data.time_interval_begin_hour == 18)), :]
    # train_data = feature_data.loc[(feature_data.time_interval_month == 4) &
    #                               (feature_data.time_interval_day < 31), :]
    train_data.loc[(train_data.time_interval_month == 4) &
                      (train_data.time_interval_day < 31) &
                      ((train_data.time_interval_begin_hour == 8) |
                       (train_data.time_interval_begin_hour == 15) |
                       (train_data.time_interval_begin_hour == 18)), 'travel_time'] = ori_feature_data.loc[
        (ori_feature_data.time_interval_month == 4) &
        (ori_feature_data.time_interval_day < 31) &
        (((ori_feature_data.time_interval_begin_hour == 8)) |
         ((ori_feature_data.time_interval_begin_hour == 15)) |
         ((ori_feature_data.time_interval_begin_hour == 18))), 'travel_time'].values
    train_history = feature_data.loc[(feature_data.time_interval_month == 3)&
                      (feature_data.time_interval_day <= 31) &
                      ((feature_data.time_interval_begin_hour == 8) |
                       (feature_data.time_interval_begin_hour == 15) |
                       (feature_data.time_interval_begin_hour == 18)), :]
    train_history = train_history.groupby(['link_ID','time_interval_begin_hour', 'time_interval_minutes'])[
        'travel_time'].agg([('median_m', np.median),
                            ('mode_m', mode_function),('quantile1_m',quantile1),
                            ('quantile2_m', quantile2),('quantile3_m',quantile3),('quantile4_m',quantile4),
                            ('quantile5_m', quantile5),('quantile6_m',quantile6),('quantile7_m',quantile7),
                            ('quantile8_m',quantile8),('quantile9_m',quantile9)]).reset_index()

    train_history['time_interval_begin_hour']  = train_history['time_interval_begin_hour'] - 1
    train_data = pd.merge(train_data, train_history, on=['link_ID','time_interval_begin_hour', 'time_interval_minutes'], how='left')

    # train_data.to_csv(data_path + '/data/LSTM_train_data_m4.csv', index=False)
    train_data = train_data.loc[:,['travel_time','median_m', 'mode_m',
                                   'quantile1_m','quantile3_m',
                                   'quantile5_m',
                                   'quantile7_m']].values



    #load test_data
    test_data_df5 = feature_data.loc[(feature_data.time_interval_month == 5)&
                                          (feature_data.time_interval_day >=15)&
                                 (
                                  (feature_data.time_interval_begin_hour == 7)|
                                  (feature_data.time_interval_begin_hour == 8)|

                                  (feature_data.time_interval_begin_hour == 14)|
                                  ((feature_data.time_interval_begin_hour == 15))|

                                  (feature_data.time_interval_begin_hour == 17)|
                                  ((feature_data.time_interval_begin_hour == 18))),:]
    test_data_df5.loc[(test_data_df5.time_interval_month == 5) &
                                 (test_data_df5.time_interval_day >= 15) &
                                 ((test_data_df5.time_interval_begin_hour == 8) |
                                  (test_data_df5.time_interval_begin_hour == 15) |
                                  (test_data_df5.time_interval_begin_hour == 18)), 'travel_time'] = ori_feature_data.loc[
                               (ori_feature_data.time_interval_month == 5) &
                                 (ori_feature_data.time_interval_day >= 15) &
                               (((ori_feature_data.time_interval_begin_hour == 8)) |
                                  ((ori_feature_data.time_interval_begin_hour == 15)) |
                                  ((ori_feature_data.time_interval_begin_hour == 18))), 'travel_time'].values

    test_history = feature_data.loc[(feature_data.time_interval_month == 4)&
                      (feature_data.time_interval_day <= 31) &
                      ((feature_data.time_interval_begin_hour == 8) |
                       (feature_data.time_interval_begin_hour == 15) |
                       (feature_data.time_interval_begin_hour == 18)), :]
    test_history = test_history.groupby(['link_ID','time_interval_begin_hour','time_interval_minutes'])[
        'travel_time'].agg([('median_m', np.median),
                            ('mode_m', mode_function), ('quantile1_m', quantile1),
                            ('quantile2_m', quantile2), ('quantile3_m', quantile3), ('quantile4_m', quantile4),
                            ('quantile5_m', quantile5), ('quantile6_m', quantile6), ('quantile7_m', quantile7),
                            ('quantile8_m', quantile8), ('quantile9_m', quantile9)]).reset_index()
    test_history['time_interval_begin_hour'] = test_history['time_interval_begin_hour'] - 1
    test_data_df5 = pd.merge(test_data_df5, test_history, on=['link_ID','time_interval_begin_hour', 'time_interval_minutes'], how='left')
    print("test_data_df5.shape",test_data_df5.shape)
    test_data5 = test_data_df5.loc[:,['travel_time','median_m', 'mode_m',
                                      'quantile1_m', 'quantile3_m',
                                      'quantile5_m',
                                      'quantile7_m']].values

    print('a',test_data5.size)


    #########6月
    test_data_df6 = feature_data.loc[(feature_data.time_interval_month == 6) &
                                     (feature_data.time_interval_day >= 0) &
                                     (
                                      (feature_data.time_interval_begin_hour == 7) |
                                      ((feature_data.time_interval_begin_hour == 8)) |

                                      (feature_data.time_interval_begin_hour == 14) |
                                      ((feature_data.time_interval_begin_hour == 15)) |

                                      (feature_data.time_interval_begin_hour == 17) |
                                      ((feature_data.time_interval_begin_hour == 18))), :]
    test_data_df6.loc[(test_data_df6.time_interval_month == 6) &
                      (test_data_df6.time_interval_day >= 0) &
                      ((test_data_df6.time_interval_begin_hour == 8) |
                       (test_data_df6.time_interval_begin_hour == 15) |
                       (test_data_df6.time_interval_begin_hour == 18)), 'travel_time'] = ori_feature_data.loc[
                                                                            (ori_feature_data.time_interval_month == 6) &
                                                                            (ori_feature_data.time_interval_day >= 0) &
                                                                            (((ori_feature_data.time_interval_begin_hour == 8)) |
                                                                             ((
                                                                              ori_feature_data.time_interval_begin_hour == 15)) |
                                                                             ((
                                                                              ori_feature_data.time_interval_begin_hour == 18))),
                                                                            'travel_time']

    test_history = feature_data.loc[(feature_data.time_interval_month == 5)&
                      (feature_data.time_interval_day <= 31) &
                      ((feature_data.time_interval_begin_hour == 8) |
                       (feature_data.time_interval_begin_hour == 15) |
                       (feature_data.time_interval_begin_hour == 18)), :]
    test_history = test_history.groupby(['link_ID', 'time_interval_begin_hour', 'time_interval_minutes'])[
        'travel_time'].agg([('median_m', np.median),
                            ('mode_m', mode_function), ('quantile1_m', quantile1),
                            ('quantile2_m', quantile2), ('quantile3_m', quantile3), ('quantile4_m', quantile4),
                            ('quantile5_m', quantile5), ('quantile6_m', quantile6), ('quantile7_m', quantile7),
                            ('quantile8_m', quantile8), ('quantile9_m', quantile9)]).reset_index()
    test_history['time_interval_begin_hour'] = test_history['time_interval_begin_hour'] - 1
    test_data_df6 = pd.merge(test_data_df6, test_history,
                             on=['link_ID', 'time_interval_begin_hour', 'time_interval_minutes'], how='left')
    test_data6 = test_data_df6.loc[:,['travel_time','median_m', 'mode_m',
                                      'quantile1_m', 'quantile3_m',
                                      'quantile5_m',
                                      'quantile7_m']].values

    print('test_data6.shape',test_data6.shape)

    ###########7月
    test_data7 = feature_data.loc[(pd.to_datetime(feature_data.time_interval_begin) > pd.to_datetime('2017-01-01')) &
                                  (feature_data.time_interval_month == 7)&
                                     (feature_data.time_interval_day <= 31) &
                                     (
                                      (feature_data.time_interval_begin_hour == 7) |
                                      ((feature_data.time_interval_begin_hour == 8)) |

                                      (feature_data.time_interval_begin_hour == 14) |
                                      ((feature_data.time_interval_begin_hour == 15)) |

                                      (feature_data.time_interval_begin_hour == 17) |
                                      ((feature_data.time_interval_begin_hour == 18))),:]
    test_data7.loc[(pd.to_datetime(test_data7.time_interval_begin) > pd.to_datetime('2017-01-01')) &
                   (test_data7.time_interval_month == 7) &
                      (test_data7.time_interval_day <= 31) &
                      ((test_data7.time_interval_begin_hour == 8) |
                       (test_data7.time_interval_begin_hour == 15) |
                       (test_data7.time_interval_begin_hour == 18)), 'travel_time'] = ori_feature_data.loc[
        (pd.to_datetime(ori_feature_data.time_interval_begin) > pd.to_datetime('2017-01-01')) &
        (ori_feature_data.time_interval_month == 7) &
        (ori_feature_data.time_interval_day <= 31) &
        (((ori_feature_data.time_interval_begin_hour == 8)) |
         ((ori_feature_data.time_interval_begin_hour == 15)) |
         ((ori_feature_data.time_interval_begin_hour == 18))), 'travel_time'].values

    test_history = feature_data.loc[(feature_data.time_interval_month == 5)&
                      (feature_data.time_interval_day <= 31) &
                      ((feature_data.time_interval_begin_hour == 8) |
                       (feature_data.time_interval_begin_hour == 15) |
                       (feature_data.time_interval_begin_hour == 18)), :]
    test_history = test_history.groupby(['link_ID', 'time_interval_begin_hour', 'time_interval_minutes'])[
        'travel_time'].agg([('median_m', np.median),
                            ('mode_m', mode_function), ('quantile1_m', quantile1),
                            ('quantile2_m', quantile2), ('quantile3_m', quantile3), ('quantile4_m', quantile4),
                            ('quantile5_m', quantile5), ('quantile6_m', quantile6), ('quantile7_m', quantile7),
                            ('quantile8_m', quantile8), ('quantile9_m', quantile9)]).reset_index()
    test_history['time_interval_begin_hour'] = test_history['time_interval_begin_hour'] - 1
    test_data7 = pd.merge(test_data7, test_history,
                             on=['link_ID', 'time_interval_begin_hour', 'time_interval_minutes'], how='left')
    test_data7 = test_data7.loc[:,['travel_time','median_m', 'mode_m',
                                   'quantile1_m','quantile3_m',
                                   'quantile5_m',
                                   'quantile7_m']].values



    # print(test_data7.shape)
    print('test_data len:',len(test_data7))
    # print('sequence len:',seq_len)

    test_label5 = feature_data.loc[(feature_data.time_interval_month == 5) &
                                 (feature_data.time_interval_day >= 15) &
                                 (((feature_data.time_interval_begin_hour == 8)) |
                                  ((feature_data.time_interval_begin_hour == 15)) |
                                  ((feature_data.time_interval_begin_hour == 18))), :]
    test_label6 = feature_data.loc[(feature_data.time_interval_month == 6) &
                                   (feature_data.time_interval_day >= 0) &
                                   ((feature_data.time_interval_begin_hour == 8) |
                                    (feature_data.time_interval_begin_hour == 15) |
                                    (feature_data.time_interval_begin_hour == 18)), :]
    test_label7 = feature_data.loc[(pd.to_datetime(feature_data.time_interval_begin) > pd.to_datetime('2017-01-01')) &(feature_data.time_interval_month == 7) &
                                   (feature_data.time_interval_day <= 31) &
                                   ((feature_data.time_interval_begin_hour == 8) |
                                    (feature_data.time_interval_begin_hour == 15) |
                                    (feature_data.time_interval_begin_hour == 18)), :]
    test_label7 = test_label7.loc[:,['link_ID','time_interval_day','time_interval_begin_hour','time_interval_minutes']]
    test_label7 = test_label7.drop_duplicates()
    print('test_label7',test_label7.shape)
    sequence_length = seq_len + pred_len
    train_result = []
    # for index in range(len(train_data) - sequence_length):
    #    train_result.append(train_data[index: index + sequence_length])  #得到长度为seq_len+1的向量，最后一个作为label
    for day in range(30*132):
         for hour in range(3):
             temp_data = train_data[(day * 6 * 30) + (2 * hour * 30):(day * 6 * 30) + (2 * hour * 30) + sequence_length]
             #print("temp_data.shape", temp_data.shape)
             if len(temp_data)<1:
                # print("ca")
                 continue
             train_result.append(temp_data)
    train_result = np.array(train_result)
    print('train')
    print('train_result len:',len(train_result))
    print('train_result shape:',train_result.shape)

    test_result5 = []
    for day in range(17*132):
        for hour in range(3):
            temp_data = test_data5[(day * 6 * 30) + (2 * hour * 30):(day * 6 * 30) + (2 * hour * 30) + sequence_length]
            #print("temp_data.shape", temp_data.shape)
            if len(temp_data)<1:
               # print("ca")
                continue
            test_result5.append(temp_data)
    test_result5 = np.array(test_result5)
    print("test_result5.shape",test_result5.shape)

    test_result6 = []
    for day in range(30 * 132):
        for hour in range(3):
            temp_data = test_data6[(day * 6 * 30) + (2 * hour * 30):(day * 6 * 30) + (2 * hour * 30) + sequence_length]
            if len(temp_data) < 1:
                continue
            test_result6.append(temp_data)
    test_result6 = np.array(test_result6)
    test_result7 = []
    for day in range(31 * 132):
        for hour in range(3):
            temp_data = test_data7[(day * 6 * 30) + (2 * hour * 30):(day * 6 * 30) + (2 * hour * 30) + sequence_length]
            if len(temp_data) < 1:
                continue
            test_result7.append(temp_data)
    test_result7 = np.array(test_result7)

    print('test')
    #划分train、test

    feature_num = 7
    x_train = train_result[:, :30, :feature_num]
    y_train = train_result[:, 30:, 0]
    print('y_train.shape',y_train.shape)
    x_test5 = test_result5[:, :30, :feature_num]
    y_test5 = test_result5[:, 30:, 0]
    x_test6 = test_result6[:, :30, :feature_num]
    y_test6 = test_result6[:, 30:, 0]
    x_test7 = test_result7[:, :30, :feature_num]
    y_test7 = test_result7[:, 30:, 0]
    # set the number of timesteps
    timesteps = 30
    x_train = np.reshape(x_train, (x_train.shape[0], timesteps, feature_num))
    x_test5 = np.reshape(x_test5, (x_test5.shape[0], timesteps, feature_num))
    x_test6 = np.reshape(x_test6, (x_test6.shape[0], timesteps, feature_num))
    x_test7 = np.reshape(x_test7, (x_test7.shape[0], timesteps, feature_num))

    return [x_train, y_train, x_test5, y_test5, x_test6, y_test6, x_test7, y_test7, test_label5, test_label6, test_label7]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:   #window shape (sequence_length L ,)  即(51L,)
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):  #layers [1,50,100,1]
    model = Sequential()

    model.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2],return_sequences=False))
    model.add(Dropout(0.2))

    # model.add(LSTM(layers[3], return_sequences=False))
    # model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers[3]))   ######结构调参
    model.add(Activation("relu"))#####ac调参       relu  linear

    start = time.time()
    model.compile(loss="mape", optimizer="adam", metrics=['mape'])     #####op调参   SGD  rmsprop
    print("Compilation Time : ", time.time() - start)
    return model

#直接全部预测
def predict_point_by_point(model, data):
    predicted = model.predict(data)
    print('predicted shape:',np.array(predicted).shape)  #(412L,1L)
    predicted = np.reshape(predicted, (predicted.size,))
    # predicted = np.reshape(predicted, (1,))
    return predicted

#滚动预测
def predict_sequence_full(model, data, window_size):  #data X_test
    curr_frame = data[0]  #(50L,1L)
    predicted = []
    for i in np.arange(len(data)):
        #x = np.array([[[1],[2],[3]], [[4],[5],[6]]])  x.shape (2, 3, 1) x[0,0] = array([1])  x[:,np.newaxis,:,:].shape  (2, 1, 3, 1)
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])  #np.array(curr_frame[newaxis,:,:]).shape (1L,50L,1L)
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)   #numpy.insert(arr, obj, values, axis=None)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):  #window_size = seq_len
    prediction_seqs = []
    for i in np.arange(len(data)/prediction_len):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in np.arange(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def plot_results(predicted_data, true_data, filename):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    plt.savefig(filename+'.png')

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in np.arange(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
    plt.savefig('plot_results_multiple.png')
from keras.utils.vis_utils import plot_model
if __name__=='__main__':
    # test()
    global_start_time = time.time()
    epochs  = 200
    seq_len = 30         #####特征序列长度
    pred_len = 30

    print('> Loading data... ')

    X_train, y_train, X_test5, y_test5, X_test6, y_test6, X_test7, y_test7, test_label5, test_label6, test_label7 = load_data(seq_len,pred_len)


    print('X_train shape:',X_train.shape)  #(3709L, 50L, 1L)
    print('y_train shape:',y_train.shape)  #(3709L,)
    print('X_test shape:',X_test5.shape)    #(412L, 50L, 1L)
    print('y_test shape:',y_test5.shape)    #(412L,)

    print('> Data Loaded. Compiling...')

    # model = build_model([1, 300,700, 30])
    ########model
    model = Sequential()

    model.add(LSTM(input_dim=7, output_dim=1000, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim = pred_len))  ######结构调参
    model.add(Activation("relu"))  #####ac调参       relu  linear  selu  tanh

    start = time.time()
    model.compile(loss="mape", optimizer="rmsprop")  #####op调参   SGD  rmsprop
    print("Compilation Time : ", time.time() - start)
    print(model.summary())
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    best_weights_filepath = './best_weights_' + str(pred_len) + '.hdf5'
    earlyStopping = kcallbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    saveBestModel = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1,
                                               save_best_only=True, mode='auto')
    model.fit(X_train,y_train,batch_size=128,nb_epoch=epochs,validation_data=[X_test5,y_test5],
              callbacks=[earlyStopping, saveBestModel])
    model.load_weights(best_weights_filepath)
    #########d
    point_by_point_predictions5 = predict_point_by_point(model, X_test5)
    point_by_point_predictions6 = predict_point_by_point(model, X_test6)
    point_by_point_predictions7 = predict_point_by_point(model, X_test7)
    ####输出预测结果
    test_label5['lstm_pred'] = point_by_point_predictions5
    test_label6['lstm_pred'] = point_by_point_predictions6
    test_label7['lstm_pred'] = point_by_point_predictions7
    test_label5.to_csv(data_path+'/data/LSTM_pred_m5.csv',index=False)
    test_label6.to_csv(data_path+'/data/LSTM_pred_m6.csv',index=False)
    test_label7.to_csv(data_path+'/data/LSTM_pred_m7.csv',index=False)
    ####
    y_test5 = np.array(y_test5)
    y_test5 = np.reshape(y_test5 ,(y_test5.size,))
    y_test6 = np.array(y_test6)
    y_test6 = np.reshape(y_test6, (y_test6.size,))
    print('Training duration (s) : ', time.time() - global_start_time)

    print('M5 mape:',mape(y_test5,point_by_point_predictions5))
    print('M6 mape:',mape(y_test6,point_by_point_predictions6))
    plot_results(point_by_point_predictions5,y_test5,'point_by_point_predictions')