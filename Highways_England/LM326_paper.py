# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from time import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def load_data(file_path, seq_len=50):
    global T_MIN, T_MAX
    df = pd.read_csv(file_path, delimiter=';', parse_dates=['date'])
    T_MIN = df['length'][0]*3.6/120
    T_MAX = df['travel_time'].max()
    # log ?
    df['travel_time_normalise'] = df['travel_time'].map(lambda x: (x-T_MIN) / (T_MAX-T_MIN))

    data = df['travel_time_normalise'].values
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    print('>>> shape of x_train (%d, %d, %d)' % x_train.shape)
    print('>>> shape of y_train (%d,)' % y_train.shape)
    # print('>>> shape of x_test (%d, %d, %d)' % x_test.shape)
    # print('>>> shape of y_test (%d,)' % y_test.shape)
    return [x_train, y_train, x_test, y_test]


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
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time() - start)
    return model


def get_mre(true, pre):
    assert len(true) == len(pre)
    return np.sum(np.abs(true-pre)/true)/len(pre)


def plot_results(true_data, predicted_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    T_MIN, T_MAX = 30, 510
    global_start_time = time()
    epochs = 50
    seq_len = 20
    features = 1
    hidden = 100
    output = 1
    print('>>> loading data... ')
    X_train, y_train, X_test, y_test = load_data('data/HE_2013_M25LM326_6_10.csv',
                                                 seq_len=seq_len)
    print('>>> Data Loaded. Compiling...')
    model = build_model(seq_len, features, hidden, output)

    # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
    history = model.fit(X_train, y_train,
                        batch_size=512,
                        epochs=epochs,
                        validation_split=0.1,
                        verbose=2)

    # predict
    predicted = model.predict(X_test)
    predicted = predicted.flatten() * (T_MAX - T_MIN) + T_MIN
    y_test = y_test * (T_MAX - T_MIN) + T_MIN
    mae = mean_absolute_error(y_test, predicted)
    print('Test MAE: %.3f' % mae)
    rmse = sqrt(mean_squared_error(y_test, predicted))
    print('Test RMSE: %.3f' % rmse)
    mre = get_mre(y_test, predicted)
    print('Test MRE: %.3f' % mre)
    print('Training duration (s) : ', time() - global_start_time)
    plot_results(y_test, predicted)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend()
    plt.show()
