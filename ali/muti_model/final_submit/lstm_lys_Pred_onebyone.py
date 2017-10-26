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
    

def load_data(seq_len):
    ori_feature_data = pd.read_csv(data_path+'/data'+'/all_data_M34567.csv',dtype = {"link_ID":str})
    #feature_data = pd.read_csv(data_path+'/data'+'/feature_data_2017_4567_replaceForHigherPoint.csv',dtype = {"link_ID":str})
    
    #feature_data = pd.read_csv(data_path+'/data'+'/quaterfinal_all_link_ori_data_0-24_replace_missval.csv',dtype = {"link_ID":str})
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
    train_history = feature_data.loc[(feature_data.time_interval_month == 3), :]
    train_history = train_history.groupby(['link_ID', 'time_interval_minutes'])[
        'travel_time'].agg([ ('median_m', np.median),
                            ('mode_m', mode_function),('std_m',np.std)]).reset_index()
    train_data = pd.merge(train_data, train_history, on=['link_ID', 'time_interval_minutes'], how='left')

    train_data = train_data.loc[:,['mean_m','median_m','mode_m','std_m','travel_time']].values

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

    test_history = feature_data.loc[(feature_data.time_interval_month == 4), :]
    test_history = test_history.groupby(['link_ID', 'time_interval_minutes'])[
        'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                            ('mode_m', mode_function),('std_m',np.std)]).reset_index()

    test_data_df5 = pd.merge(test_data_df5, test_history, on=['link_ID', 'time_interval_minutes'], how='left')
    print("test_data_df5.shape",test_data_df5.shape)
    test_data5 = test_data_df5.loc[:,['mean_m','median_m','mode_m','std_m','travel_time']].values
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

    test_history = feature_data.loc[(feature_data.time_interval_month == 5), :]
    test_history = test_history.groupby(['link_ID', 'time_interval_minutes'])[
        'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                            ('mode_m', mode_function),('std_m',np.std)]).reset_index()

    test_data_df6 = pd.merge(test_data_df6, test_history, on=['link_ID', 'time_interval_minutes'], how='left')
    test_data6 = test_data_df6.loc[:,['mean_m','median_m','mode_m','std_m','travel_time']].values

    ###########7月
    test_data7 = feature_data.loc[(pd.to_datetime(feature_data.time_interval_begin) > pd.to_datetime('2017-01-01')) &
                                  (feature_data.time_interval_month == 7)&
                                     (feature_data.time_interval_day <= 15) &
                                     (
                                      (feature_data.time_interval_begin_hour == 7) |
                                      ((feature_data.time_interval_begin_hour == 8)) |

                                      (feature_data.time_interval_begin_hour == 14) |
                                      ((feature_data.time_interval_begin_hour == 15)) |

                                      (feature_data.time_interval_begin_hour == 17) |
                                      ((feature_data.time_interval_begin_hour == 18))),:]

    test_history = feature_data.loc[(feature_data.time_interval_month == 5), :]
    test_history = test_history.groupby(['link_ID', 'time_interval_minutes'])[
        'travel_time'].agg([('mean_m', np.mean), ('median_m', np.median),
                            ('mode_m', mode_function),('std_m',np.std)]).reset_index()

    test_data7 = pd.merge(test_data7, test_history, on=['link_ID', 'time_interval_minutes'], how='left')
    test_data7 = test_data7.loc[:,['mean_m','median_m','mode_m','std_m','travel_time']].values



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
                                   (feature_data.time_interval_day <= 15) &
                                   ((feature_data.time_interval_begin_hour == 8) |
                                    (feature_data.time_interval_begin_hour == 15) |
                                    (feature_data.time_interval_begin_hour == 18)), :]
    test_label7 = test_label7.loc[:,['link_ID','time_interval_day','time_interval_begin_hour','time_interval_minutes']]
    test_label7 = test_label7.drop_duplicates()
    print('test_label7',test_label7.shape)
    sequence_length = seq_len + 30
    train_result = []
#    for index in range(len(train_data) - sequence_length):
#        train_result.append(train_data[index: index + sequence_length])  #得到长度为seq_len+1的向量，最后一个作为label
    for day in range(30*132):
         for hour in range(3):
             temp_data = train_data[(day * 6 * 30) + (2 * hour * 30):(day * 6 * 30) + (2 * hour * 30) + 60]
             #print("temp_data.shape", temp_data.shape)
             if len(temp_data)<1:
                # print("ca")
                 continue
             train_result.append(temp_data)
    train_result = np.array(train_result)
    print('train')
    print('train_result len:',len(train_result))
    print('train_result shape:',train_result.shape)
    #
    # if normalise_window:
    #     result = normalise_windows(result)
    #
    # print(result[:1])
    # print('normalise_windows result shape:',np.array(result).shape)

    test_result5 = []
    for day in range(17*132):
        for hour in range(3):
            temp_data = test_data5[(day * 6 * 30) + (2 * hour * 30):(day * 6 * 30) + (2 * hour * 30) + 60]
            #print("temp_data.shape", temp_data.shape)
            if len(temp_data)<1:
               # print("ca")
                continue
            test_result5.append(temp_data)
    test_result5 = np.array(test_result5)
    print("test_result5.shape",test_result5.shape)

    # for index in range(len(test_data5) - sequence_length):
    #     test_result5.append(test_data5[index: index + sequence_length])  #得到长度为seq_len+1的向量，最后一个作为label
    # test_result5 = np.array(test_result5)


    test_result6 = []
    for day in range(30 * 132):
        for hour in range(3):
            temp_data = test_data6[(day * 6 * 30) + (2 * hour * 30):(day * 6 * 30) + (2 * hour * 30) + 60]
            if len(temp_data) < 1:
                continue
            test_result6.append(temp_data)
    test_result6 = np.array(test_result6)
    test_result7 = []
    for day in range(15 * 132):
        for hour in range(3):
            temp_data = test_data7[(day * 6 * 30) + (2 * hour * 30):(day * 6 * 30) + (2 * hour * 30) + 60]
            if len(temp_data) < 1:
                continue
            test_result7.append(temp_data)
    test_result7 = np.array(test_result7)
    # for index in range(len(test_data6) - sequence_length):
    #     test_result6.append(test_data6[index: index + sequence_length])  #得到长度为seq_len+1的向量，最后一个作为label
    # test_result6 = np.array(test_result6)

    print('test')
    #划分train、test
    # row = round(0.9 * result.shape[0])
    # train = result[:row, :]
    # np.random.shuffle(train)

    #####for new model
    # y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    # y_test5 = np.reshape(y_test5, (y_test5.shape[0], y_test5.shape[1], 1))
    # y_test6 = np.reshape(y_test6, (y_test6.shape[0], y_test6.shape[1], 1))
    # y_test7 = np.reshape(y_test7, (y_test7.shape[0], y_test7.shape[1], 1))
    #######for ori_model
    # y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))
    # y_test5 = np.reshape(y_test5, (y_test5.shape[0], y_test5.shape[1]))
    # y_test6 = np.reshape(y_test6, (y_test6.shape[0], y_test6.shape[1]))
    # y_test7 = np.reshape(y_test7, (y_test7.shape[0], y_test7.shape[1]))


    # x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    # x_test5 = np.reshape(x_test5, (x_test5.shape[0], 1, x_test5.shape[1]))
    # # y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
    # # y_test5 = np.reshape(y_test5, (y_test5.shape[0], 1, y_test5.shape[1]))
    # x_test6 = np.reshape(x_test6, (x_test6.shape[0], 1, x_test6.shape[1]))
    # x_test7 = np.reshape(x_test7, (x_test7.shape[0], 1, x_test7.shape[1]))
    # y_test6 = np.reshape(y_test6, (y_test6.shape[0], 1, y_test6.shape[1]))
    #return [x_train, y_train, x_test5, y_test5, x_test6, y_test6, x_test7, y_test7, test_label5, test_label6, test_label7]
    return [train_result, test_result5 ,test_result6, test_result7,  test_label5, test_label6, test_label7]
    
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

#mape评测函数    
def mape1(y,d):
    c=d
    result= -np.sum(np.abs(y-c)/c)/len(c)
    return "mape",result
    
    
from keras.utils.vis_utils import plot_model
if __name__=='__main__':
    # test()
    global_start_time = time.time()
    epochs  = 50
    seq_len = 30         #####特征序列长度

    print('> Loading data... ')

    train_result, test_result5 ,test_result6, test_result7 , test_label5, test_label6, test_label7= load_data(seq_len)
    #X_train, y_train, X_test5, y_test5, X_test6, y_test6, X_test7, y_test7, test_label5, test_label6, test_label7 = load_data(seq_len)
    
    #X
    X_train = train_result[:, :30, :5]
    X_test5 = test_result5[:, :30, :5]
    X_test6 = test_result6[:, :30, :5]
    X_test7 = test_result7[:, :30, :5]
        
    # set the number of timesteps
    timesteps = 30
    # reshape input to be: (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], timesteps, 5))
    X_test5 = np.reshape(X_test5, (X_test5.shape[0], timesteps, 5))
    X_test6 = np.reshape(X_test6, (X_test6.shape[0], timesteps, 5))
    X_test7 = np.reshape(X_test7, (X_test7.shape[0], timesteps, 5))
    
    #y
    for label_lengh in range(1,29,1):
            print("")
            print("")
            print("########第",label_lengh,"轮")
            y_train = train_result[:, 30:30+label_lengh, 4]
            y_test5 = test_result5[:, 30:30+label_lengh, 4]
            y_test6 = test_result6[:, 30:30+label_lengh, 4]
            y_test7 = test_result7[:, 30:30+label_lengh, 4]    
        
            print('X_train shape:',X_train.shape)  #(3709L, 50L, 1L)
            print('y_train shape:',y_train.shape)  #(3709L,)
            print('X_test shape:',X_test5.shape)    #(412L, 50L, 1L)
            print('y_test shape:',y_test5.shape)    #(412L,)
        
        
        
            print('> Data Loaded. Compiling...')
        
            # model = build_model([1, 300,700, 30])
            ########model
            model = Sequential()
        
            model.add(LSTM(input_dim=5, output_dim=2000, return_sequences=False))
            model.add(Dropout(0.2))
            # model.add(LSTM(200, return_sequences=True))
            # model.add(LSTM(300, return_sequences=True))
            # model.add(LSTM(400, return_sequences=True))
            # model.add(LSTM(500, return_sequences=True))
            # model.add(LSTM(600, return_sequences=True))
            # model.add(LSTM(700, return_sequences=True))
            # model.add(LSTM(800, return_sequences=True))
            # model.add(LSTM(900, return_sequences=True))
#            model.add(LSTM(1000, return_sequences=False))
#            model.add(Dropout(0.2))
            # model.add(LSTM(output_dim=100, return_sequences=False))
            # model.add(LSTM(20,input_shape=(30,3), return_sequences=True))
            # model.add(LSTM(20,input_shape=(30,3), return_sequences=True))
            # model.add(Dropout(0.4))
        
            # model.add(LSTM(layers[2], return_sequences=False))
            # model.add(Dropout(0.2))
        
            # model.add(LSTM(layers[3], return_sequences=False))
            # model.add(Dropout(0.2))
        
            model.add(Dense(output_dim =label_lengh ))  ######结构调参
            model.add(Activation("relu"))  #####ac调参       relu  linear
        
            start = time.time()
            model.compile(loss="mape", optimizer="rmsprop")  #####op调参   SGD  rmsprop
            print("Compilation Time : ", time.time() - start)
            print(model.summary())
            # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
            
            best_weights_filepath = './best_weights_'+str(label_lengh)+'.hdf5'
            earlyStopping=kcallbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')
            saveBestModel = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        
            # train model
            history = model.fit(X_train, y_train, batch_size=128, nb_epoch=epochs,
                      verbose=1, validation_data=(X_test5,y_test5), callbacks=[earlyStopping, saveBestModel])
        
            #reload best weights
            model.load_weights(best_weights_filepath)
        
            #########d
        
            # multiple_predictions = predict_sequences_multiple(model, X_test, seq_len, prediction_len=50)
            # print('multiple_predictions shape:',np.array(multiple_predictions).shape)   #(8L,50L)
            #
            # full_predictions = predict_sequence_full(model, X_test, seq_len)
            # print('full_predictions shape:',np.array(full_predictions).shape)    #(412L,)
        
            point_by_point_predictions5 = predict_point_by_point(model, X_test5)
            point_by_point_predictions6 = predict_point_by_point(model, X_test6)
            point_by_point_predictions7 = predict_point_by_point(model, X_test7)
            
            
            ####输出预测结果
            #构造输出结果的前缀
            test_label5_tmp = test_label5.loc[test_label5.time_interval_minutes<2*label_lengh]
            test_label6_tmp = test_label6.loc[test_label6.time_interval_minutes<2*label_lengh]
            test_label7_tmp = test_label7.loc[test_label7.time_interval_minutes<2*label_lengh]  
            
            test_label5_tmp['lstm_pred'+str(label_lengh)] = point_by_point_predictions5
            test_label6_tmp['lstm_pred'+str(label_lengh)] = point_by_point_predictions6
            test_label7_tmp['lstm_pred'+str(label_lengh)] = point_by_point_predictions7
            test_label5_tmp.to_csv(data_path+'/data/jiaocha/LSTM_pred_m5_'+str(label_lengh)+'.csv',index=False)
            test_label6_tmp.to_csv(data_path+'/data/jiaocha/LSTM_pred_m6_'+str(label_lengh)+'.csv',index=False)
            test_label7_tmp.to_csv(data_path+'/data/jiaocha/LSTM_pred_m7_'+str(label_lengh)+'.csv',index=False)
            ####
            # print('point_by_point_predictions shape:',np.array(point_by_point_predictions).shape)  #(412L)
            y_test50 = np.array(y_test5)
            y_test50 = np.reshape(y_test50 ,(y_test50.size,))
            y_test60 = np.array(y_test6)
            y_test60 = np.reshape(y_test60, (y_test60.size,))
            print('Training duration (s) : ', time.time() - global_start_time)
        
            # plot_results_multiple(multiple_predictions, y_test, 50)
            # plot_results(full_predictions,y_test,'full_predictions')
            print('M5 mape:',mape(y_test50,point_by_point_predictions5))
            print('M6 mape:',mape(y_test60,point_by_point_predictions6))
            #plot_results(point_by_point_predictions5,y_test5,'point_by_point_predictions')
    
    
    
    