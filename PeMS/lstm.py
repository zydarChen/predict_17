# -*- coding: utf-8 -*-
from numpy.random import seed
seed(1717)
from tensorflow import set_random_seed
set_random_seed(1717)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.allow_growth = True
sess = tf.Session(graph=tf.get_default_graph(), config=config)
from keras import backend as K
K.set_session(sess)

import gc
from time import time
from sklearn.preprocessing import StandardScaler
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.core import Dense
from PeMS.utils import *
from PeMS.pre_weather import weather_dataframe


def series2supervised(data, timesteps=20, size=(1, 2)):
    assert isinstance(size[0], int) & isinstance(size[1], int), 'size[0] size[1] must be integer'
    assert size[0] <= size[1], 'size[1] not less than size[0]'
    start = '2016-%02d' % size[0]
    end = '2016-%02d' % size[1]
    value = data[start: end].copy().values
    result = []
    for index in range(len(value) - timesteps):
        result.append(value[index: index + timesteps + 1])
    result = np.array(result)
    x = result[:, :-1]
    y = result[:, -1]
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y


def multi_series2supervised(data, timesteps=20, size=(1, 2), dropnan=True, year=2016):
    """
    :param data: DataFrame, data.index.dtype=datetime64, first columns is flow
    :param timesteps: int
    :param size: tuple, (start_month, end_month)
    :param dropnan: bool
    :param year: int
    :return: x.shape=(sample, timesteps, feature), y.shape=(sample, flow)
    """
    assert isinstance(size[0], int) & isinstance(size[1], int), 'size[0] size[1] must be integer'
    assert size[0] <= size[1], 'size[1] not less than size[0]'
    start = '%d-%02d' % (year, size[0])
    end = '%d-%02d' % (year, size[1])
    df = data[start: end].copy()
    old_names = df.columns
    n_vars = len(old_names)
    cols, names = list(), list()
    for i in range(timesteps, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (old_names[j], i)) for j in range(n_vars)]
    cols.append(df[old_names[0]])
    names.append(old_names[0])
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    x = agg.values[:, :-1]
    x = x.reshape(x.shape[0], timesteps, n_vars)
    y = agg.values[:, -1]
    return x, y


def build_model(timestep=20, features=1, hidden=100, output=1,
                dropout=0.5, activation='linear', loss='mape', optimizer='rmsprop'):
    model = Sequential()

    model.add(LSTM(
        input_shape=(timestep, features),
        units=timestep,
        dropout=dropout,
        return_sequences=True))

    model.add(LSTM(
        units=hidden,
        dropout=dropout,
        return_sequences=False))

    model.add(Dense(
        units=output,
        activation=activation))

    start = time()
    model.compile(loss=loss, optimizer=optimizer)
    print('>>> Compilation Time : ', time() - start)
    return model


def get_train_test(flow_path='./data/SR99_VDS1005210_2016_fill.csv'):
    df = pd.read_csv(flow_path, delimiter=';', parse_dates=True, index_col='datetime')
    interval = 5
    timesteps = 50
    epochs = 100
    batch_size = int(60 / interval * 24 * 7)  # 每批处理一周数据

    train = df[: '2016-10'].copy()
    test = df['2016-11': '2016-12'].copy()
    scaler = StandardScaler()
    train['flow_scaler'] = scaler.fit_transform(train['flow_5'].values[:, np.newaxis])
    test['flow_scaler'] = scaler.transform(test['flow_5'].values[:, np.newaxis])
    x_train, y_train = series2supervised(train['flow_scaler'], timesteps=timesteps, size=(1, 8))
    x_valid, y_valid = series2supervised(train['flow_scaler'], timesteps=timesteps, size=(9, 10))
    x_test, y_test = series2supervised(test['flow_scaler'], timesteps=timesteps, size=(11, 12))

    model = Sequential()
    model.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                        epochs=epochs, batch_size=batch_size, verbose=2)
    # predict
    predict = model.predict(x_test)
    predict = scaler.inverse_transform(predict).flatten()
    expect = scaler.inverse_transform(y_test).flatten()
    print_error(expect, predict)
    plot_results(expect[1000:1288], predict[1000:1288])
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend()
    plt.savefig('../res/loss.svg', format='svg')
    plt.show()
    # 解决Exception ignored in BaseSession
    gc.collect()


def cross_validation():
    """
    CV
    [1, 5] predict 6
    [1, 6] predict 7
    [1, 7] predict 8
    [1, 8] predict 9
    [1, 9] predict 10
    test
    [1, 10] predict [11, 12]
    :return:
    """
    pass


def get_train_test_with_weather(log=-121.13333, lat=37.746753):
    df_pems = pd.read_csv('./data/SR99_VDS1005210_2016_fill.csv', delimiter=';', parse_dates=True, index_col='datetime')
    df_weather = weather_dataframe(log, lat)
    df_weather = df_weather[['datetime', 'temp', 'visibility']].set_index('datetime')
    df = pd.merge(df_pems, df_weather, left_index=True, right_index=True, how='outer').fillna(method='ffill').fillna(method='bfill')

    interval = 5
    timesteps = 50
    epochs = 100
    batch_size = int(60 / interval * 24 * 7)  # 每批处理一周数据

    train = df[: '2016-10'].copy()
    test = df['2016-11': '2016-12'].copy()
    col_names = test.columns
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train.values)
    scaled_test = scaler.transform(test.values)
    for i in range(len(col_names)):
        train[col_names[i]] = scaled_train[:, i]
        test[col_names[i]] = scaled_test[:, i]

    x_train, y_train = multi_series2supervised(train, timesteps=timesteps, size=(1, 8))
    x_valid, y_valid = multi_series2supervised(train, timesteps=timesteps, size=(9, 10))
    x_test, y_test = multi_series2supervised(test, timesteps=timesteps, size=(11, 12))

    model = Sequential()
    model.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                        epochs=epochs, batch_size=batch_size, verbose=2)
    # predict
    zero_col = np.zeros((y_test.shape[0], 2), dtype='float32')
    predict = model.predict(x_test)
    predict = scaler.inverse_transform(np.concatenate((predict, zero_col), axis=1))[:, 0]
    expect = scaler.inverse_transform(np.concatenate((y_test[:, np.newaxis], zero_col), axis=1))[:, 0]
    print_error(expect, predict)
    plot_results(expect[1000:1288], predict[1000:1288])
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend()
    plt.savefig('../res/loss.svg', format='svg')
    plt.show()
    # 解决Exception ignored in BaseSession
    gc.collect()


if __name__ == '__main__':
    get_train_test()
    # get_train_test_with_weather()



