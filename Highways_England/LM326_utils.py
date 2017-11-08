# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import backend as K
from time import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

T_MIN, T_MAX = 0, 0


def mre(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true)/y_true, axis=-1)


def loss_paper(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)/2


def get_mre(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.mean(np.abs(y_true-y_pred)/y_true)


def load_data(file_path, seq_len=50, norm=1):
    global T_MIN, T_MAX
    scaler = StandardScaler()
    df = pd.read_csv(file_path, delimiter=';', parse_dates=['date'])
    T_MIN = df['length'][0]*3.6/120
    T_MAX = df['travel_time'].max()
    # log ?
    if norm == 0:  # 不正则化
        data = df['travel_time'].values
    elif norm == 1:  # MAX-MIN
        df['travel_time_normalise'] = df['travel_time'].map(lambda x: (x - T_MIN) / (T_MAX - T_MIN))
        data = df['travel_time_normalise'].values
    else:  # 标准化
        values = df['travel_time'].values
        data = scaler.fit_transform(values[:, np.newaxis]).flatten()
        print('>>> scaler.mean_ = %f, scaler.var_ = %f' % (scaler.mean_, scaler.var_))

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length + 1):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row_8 = round(0.8 * result.shape[0])
    row_9 = round(0.9 * result.shape[0])
    train = result[:int(row_8), :]
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row_9):, :-1]
    y_test = result[int(row_9):, -1]
    x_valid = result[int(row_8):int(row_9), :-1]
    y_valid = result[int(row_8):int(row_9), -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))

    print('>>> shape of x_train (%d, %d, %d)' % x_train.shape)
    print('>>> shape of y_train (%d,)' % y_train.shape)
    # print('>>> shape of x_test (%d, %d, %d)' % x_test.shape)
    # print('>>> shape of y_test (%d,)' % y_test.shape)
    return [scaler, x_train, y_train, x_valid, y_valid, x_test, y_test]


def de_norm(lst, norm=1, scaler=None):
    if norm == 0:
        return lst
    elif norm == 1:
        return lst * (T_MAX - T_MIN) + T_MIN
    else:
        return scaler.inverse_transform(lst)


def build_model(timestep=50, features=1, hidden=100, output=1):
    model = Sequential()

    model.add(LSTM(
        input_shape=(timestep, features),
        units=timestep,
        return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(
        units=hidden,
        return_sequences=False))
    model.add(Dropout(0.5))

    model.add(Dense(
        units=output))
    model.add(Activation("linear"))

    start = time()
    model.compile(loss=loss_paper, optimizer="rmsprop")
    print("> Compilation Time : ", time() - start)
    return model


def plot_results(true_data, predicted_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def predict_sequence_full(model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted
