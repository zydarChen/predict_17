# -*- coding: utf-8 -*-

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Highways_England.LM326_utils import *


if __name__ == '__main__':
    global_start_time = time()
    epochs = 100
    seq_len = 20
    features = 1
    hidden = 100
    output = 1
    norm = 2
    print('>>> loading data... ')
    scaler, X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
        'data/HE_2013_M25LM326_6_10.csv', seq_len=seq_len, norm=norm)
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
    predicted = de_norm(predicted.flatten(), norm=norm, scaler=scaler)
    y_test = de_norm(y_test, norm=norm, scaler=scaler)
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
