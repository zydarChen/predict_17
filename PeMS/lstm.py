# -*- coding: utf-8 -*-
import gc
from time import time
from sklearn.preprocessing import StandardScaler
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.core import Dense
from PeMS.utils import *


def series2supervised(data, time_steps=20, size=(1, 2)):
    assert isinstance(size[0], int) & isinstance(size[1], int), 'size[0] size[1] must be integer'
    assert size[0] <= size[1], 'size[1] not less than size[0]'
    start = '2016-0%d' % size[0]
    end = '2016-0%d' % size[1]
    value = data[start: end].copy().values
    result = []
    for index in range(len(value) - time_steps):
        result.append(value[index: index+time_steps+1])
    result = np.array(result)
    x = result[:, :-1]
    y = result[:, -1]
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
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


if __name__ == '__main__':
    df = pd.read_csv('./data/SR99_VDS1005210_2016_fill.csv',
                     delimiter=';', parse_dates=True, index_col='datetime')
    # df = df.resample('45min', closed='right', label='right').sum()
    # 参数定义
    interval = 5
    time_steps = 50
    epochs = 100

    batch_size = int(60 / interval * 24 * 7)  # 每批处理一周数据
    train = df[: '2016-10'].copy()
    test = df['2016-11':].copy()
    scaler = StandardScaler()
    train['flow_scaler'] = scaler.fit_transform(train['flow_5'].values[:, np.newaxis])
    test['flow_scaler'] = scaler.transform(test['flow_5'].values[:, np.newaxis])
    x_train, y_train = series2supervised(train['flow_scaler'], time_steps=time_steps, size=(1, 8))
    x_valid, y_valid = series2supervised(train['flow_scaler'], time_steps=time_steps, size=(9, 10))
    x_test, y_test = series2supervised(test['flow_scaler'], time_steps=time_steps, size=(11, 12))
    # build model
    # model = build_model(timestep=x_train.shape[1], features=x_train.shape[2], loss='mae', optimizer='adam')

    model = Sequential()
    model.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                        epochs=epochs, batch_size=batch_size, verbose=1)
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




