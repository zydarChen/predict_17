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
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def mre(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true)/y_true, axis=-1)


def get_mre(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.mean(np.abs(y_true-y_pred)/y_true)


def load_data(file_path, seq_len=50):
    scaler = StandardScaler()
    df = pd.read_csv(file_path, delimiter=';', parse_dates=['date'])
    data = df['travel_time'].values
    row_9 = round(0.9 * data.shape[0])
    train = data[:int(row_9)]



    sequence_length = seq_len + 1




    result = []
    for index in range(len(data) - sequence_length):
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

    values = df['travel_time'].values
    scaled = scaler.fit_transform(values[:, np.newaxis])
    data = scaled.flatten()

    print('>>> shape of x_train (%d, %d, %d)' % x_train.shape)
    print('>>> shape of y_train (%d,)' % y_train.shape)
    # print('>>> shape of x_test (%d, %d, %d)' % x_test.shape)
    # print('>>> shape of y_test (%d,)' % y_test.shape)
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]


def de_norm(lst):
    return lst * (T_MAX - T_MIN) + T_MIN


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
    model.compile(loss='mse', optimizer="rmsprop")
    print("> Compilation Time : ", time() - start)
    return model


def plot_results(true_data, predicted_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    global_start_time = time()
    epochs = 20
    seq_len = 20
    features = 1
    hidden = 100
    output = 1
    print('>>> loading data... ')
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
        'data/HE_2013_M25LM326_6_10.csv', seq_len=seq_len, norm=2)
    print('>>> Data Loaded. Compiling...')
    model = build_model(seq_len, features, hidden, output)

    # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
    history = model.fit(X_train, y_train,
                        batch_size=512,
                        epochs=epochs,
                        validation_data=(X_valid, y_valid),
                        verbose=2)

    # predict
    predicted = model.predict(X_test)
    predicted = de_norm(predicted.flatten())
    y_test = de_norm(y_test)
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