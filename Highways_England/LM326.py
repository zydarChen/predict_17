# -*- coding: utf-8 -*-

import pandas as pd
from time import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from matplotlib import pyplot


def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=';', parse_dates=['date'])
    length = df['length'][0]
    v_max = 120
    t_min = length/v_max * 3.6
    t_max = df['travel_time'].max()
    # log ?
    df['travel_time_normalise'] = df['travel_time'].map(lambda x: (x-t_min) / (t_max+t_min))
    return df[['travel_time_normalise']]


def series_to_supervised(data, n_in=50, n_out=1):
    """转换数据形式
    Args:
        data: DataFrame
        n_in: int, 输入时间序列长度 (t-n, ... t-1)
        n_out: int, 预测时间序列长度 (t, t+1, ..., t+n)
    Returns:
        DataFrame
    """
    n_vars = data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    df_concat = pd.concat(cols, axis=1)
    df_concat.columns = names
    return df_concat


def train_test_split(data, train_size=0.8, valid_size=0.1, test_size=0.1):
    """训练集80、验证集10、测试集10
    Args:
        data: DataFrame,
        train_size: float,
        valid_size: float,
        test_size: float,
    Returns:

    """
    values = data.values
    size = data.shape[0]
    train = values[:int(size*train_size), :]
    valid = values[int(size*train_size):int(size*(train_size+valid_size)), :]
    test = values[int(size*(1-test_size)):, :]

    train_X, train_y = train[:, :-1], train[:, -1]
    valid_X, valid_y = valid[:, :-1], valid[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape [samples, time_steps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return train_X, train_y, valid_X, valid_y, test_X, test_y


if __name__ == '__main__':
    global_start_time = time()
    epochs = 1
    seq_len = 96
    print('>>> loading data... ')
    df_data = load_data('data/HE_2013_M25LM326_6_10.csv')
    df_reframe = series_to_supervised(df_data, 1, 1)
    print(df_reframe.head())
    train_X, train_y, valid_X, valid_y, test_X, test_y = train_test_split(df_reframe)

    # LSTM
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))  # shape=(*, 1, 1)
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='rmsprop')

    # fit
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(valid_X, valid_y), verbose=2,
                        shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    #
    # # make a prediction
    # yhat = model.predict(test_X)
    # test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # # invert scaling for forecast
    # inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    # inv_yhat = scaler.inverse_transform(inv_yhat)
    # inv_yhat = inv_yhat[:, 0]
    # # invert scaling for actual
    # test_y = test_y.reshape((len(test_y), 1))
    # inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    # inv_y = scaler.inverse_transform(inv_y)
    # inv_y = inv_y[:, 0]
    # # calculate RMSE
    # rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    # print('Test RMSE: %.3f' % rmse)

