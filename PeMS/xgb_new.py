# -*- coding: utf-8 -*-

import xgboost as xgb
from sklearn.model_selection import ParameterGrid
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 数据平滑
VACATION = ['2016-01-01', '2016-01-18', '2016-02-15',
            '2016-05-30', '2016-07-04', '2016-09-05',
            '2016-10-10', '2016-11-11', '2016-11-24',
            '2016-11-25', '2016-12-23', '2016-12-26',
            ]


def xgb_mape(y_predict, dtrain):
    """
    修改xgb中的损失函数为mape
    :param y_predict:
    :param dtrain:
    :return:
    """
    y_true = dtrain.get_label()
    return 'mape', np.mean(np.abs(y_true - y_predict)/np.abs(y_true)) * 100


def print_error(y_true, y_predict):
    print('MSE: %.3f' % mse(y_true, y_predict))
    print('RMSE: %.3f' % rmse(y_true, y_predict))
    print('MAE: %.3f' % mae(y_true, y_predict))
    print('MAPE: %.3f%%' % mape(y_true, y_predict))


def mse(y_true, y_predict):
    # Mean Square Error
    assert len(y_true) == len(y_predict)
    return np.square(np.mean((y_true - y_predict) ** 2))


def rmse(y_true, y_predict):
    # Root Mean Square Error
    assert len(y_true) == len(y_predict)
    return np.mean((y_true - y_predict) ** 2)


def mae(y_true, y_predict):
    # Mean Absolute Error
    assert len(y_true) == len(y_predict)
    return np.mean(np.abs(y_true - y_predict))


def mape(y_true, y_predict):
    # Mean Absolute Percentage Error
    assert len(y_true) == len(y_predict)
    return np.mean(np.abs(y_true - y_predict)/np.abs(y_true)) * 100


def feature_vis(model, feature):
    """
    特征可视化
    :param model:
    :param feature:
    :return:
    """
    length = len(feature)
    model_feature_values = model.feature_importances_
    assert len(model_feature_values) == length
    rank = np.argsort(-model_feature_values)  # np.argsort 从小到大的索引值
    rank_feature = [feature[i] for i in rank]
    plt.figure(figsize=(20, 10))
    plt.title('Feature Importance')
    plt.bar(range(length), model_feature_values[rank])
    plt.xticks(range(length), rank_feature, rotation=90)
    plt.savefig('../res/feature_vis.png', format='png')
    # plt.show()


def create_lagging(df, lagging):
    """
    创建lagging特征
    :param df: DataFrame,
    :param lagging: int,
    :return: DataFrame
    """
    cols, names = list(), list()
    for i in range(lagging, -1, -1):
        cols.append(df.shift(i))
        names.append('lagging%d' % i)
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    return agg


def is_vacation(date):
    result = 0
    if '2016-%02d-%02d' % (date.month, date.day) in VACATION:
        result = 1
    return result


def create_feature(df, lagging):
    """
    特征工程
    :param df: DataFrame
    :param lagging: int
    :return: DataFrame
    """
    df_lagging = create_lagging(df, lagging)
    # minute_of_hour
    df_lagging['minute'] = df.index.map(lambda date: date.minute)
    # hour_of_day
    df_lagging['hour'] = df.index.map(lambda date: date.hour)
    # day_of_week
    df_lagging['day_of_week'] = df.index.map(lambda date: date.weekday() + 1)
    # day_of_month
    # df_lagging['day'] = df.index.map(lambda date: date.day)
    # month_of_year
    # df_lagging['month'] = df.index.map(lambda date: date.month)
    # vacation
    df_lagging['vacation'] = df.index.map(is_vacation)
    # last_week
    # df_lagging['last_week'] = df.shift(2016)
    # combination feature
    # week划分1，234，5，67
    # hour划分012，345，678，9101112，131415，161718，192021，2223
    df_lagging.loc[df_lagging['day_of_week'].isin([1]), 'day_of_week_en'] = 1
    df_lagging.loc[df_lagging['day_of_week'].isin([2, 3, 4]), 'day_of_week_en'] = 2
    df_lagging.loc[df_lagging['day_of_week'].isin([5]), 'day_of_week_en'] = 3
    df_lagging.loc[df_lagging['day_of_week'].isin([6, 7]), 'day_of_week_en'] = 4

    df_lagging.loc[df_lagging['hour'].isin([0, 1, 2]), 'hour_en'] = 1
    df_lagging.loc[df_lagging['hour'].isin([3, 4, 5]), 'hour_en'] = 2
    df_lagging.loc[df_lagging['hour'].isin([6, 7, 8]), 'hour_en'] = 3
    df_lagging.loc[df_lagging['hour'].isin([9, 10, 11, 12]), 'hour_en'] = 4
    df_lagging.loc[df_lagging['hour'].isin([13, 14, 15]), 'hour_en'] = 5
    df_lagging.loc[df_lagging['hour'].isin([16, 17, 18]), 'hour_en'] = 6
    df_lagging.loc[df_lagging['hour'].isin([19, 20, 21]), 'hour_en'] = 7
    df_lagging.loc[df_lagging['hour'].isin([22, 23]), 'hour_en'] = 8

    # df_lagging.loc[df_lagging['hour'].isin([0, 1, 2, 3, 4, 5]), 'hour_en'] = 1
    # df_lagging.loc[df_lagging['hour'].isin([6, 7, 8]), 'hour_en'] = 2
    # df_lagging.loc[df_lagging['hour'].isin([9, 10, 11, 12]), 'hour_en'] = 3
    # df_lagging.loc[df_lagging['hour'].isin([13, 14, 15]), 'hour_en'] = 4
    # df_lagging.loc[df_lagging['hour'].isin([16, 17, 18]), 'hour_en'] = 5
    # df_lagging.loc[df_lagging['hour'].isin([19, 20, 21, 22, 23]), 'hour_en'] = 6

    # df_lagging['week_hour'] = df_lagging['day_of_week_en'].astype('str') + ',' + df_lagging['hour_en'].astype('str')
    # one-hot编码
    df_lagging = pd.get_dummies(df_lagging, columns=['day_of_week', 'hour', 'minute', 'hour_en', 'day_of_week_en'])
    # df_lagging = df_lagging.drop(['day_of_week', 'hour'], axis=1)
    return df_lagging


def train(train, eval, params, verbose=False):
    # print('training...')
    train_X = train.drop('lagging0', axis=1).values
    train_y = train['lagging0'].values
    eval_X = eval.drop('lagging0', axis=1).values
    eval_y = eval['lagging0'].values
    xgb_regressor = xgb.XGBRegressor(nthread=30, learning_rate=params['learning_rate'], objective='reg:linear',
                                     n_estimators=params['n_estimators'])
    # early_stopping_rounds 多少个迭代性能没有提升停止迭代
    xgb_regressor.fit(train_X, train_y, verbose=verbose,
                      eval_set=[(eval_X, eval_y)], eval_metric=xgb_mape, early_stopping_rounds=50)
    # param = {'max_depth': 2, 'eta': 1, 'silent': 1}
    # watchlist = [((eval_y, eval_X), 'eval'), ((train_y, train_X), 'train')]
    # num_round = 2
    # xgb_regressor = xgb.train(param, (train_y, train_X), num_round, watchlist, xgb_mapeobj, xgb_mape)
    return xgb_regressor


def cross_validation(df, params, save=False, vis=False, verbose=False):
    xgb_regressor1 = train(df['2016-01': '2016-08'], df['2016-09': '2016-10'], params, verbose=verbose)
    xgb_regressor2 = train(df['2016-01': '2016-04'], df['2016-05': '2016-06'], params, verbose=verbose)
    xgb_regressor3 = train(df['2016-01': '2016-06'], df['2016-07': '2016-08'], params, verbose=verbose)
    xgb_regressor4 = train(df['2016-01': '2016-08'], df['2016-09': '2016-10'], params, verbose=verbose)
    train_mape = [xgb_regressor1.best_score, xgb_regressor2.best_score,
                  xgb_regressor3.best_score, xgb_regressor4.best_score]
    ###
    # 训练集测试集的划分是否需要shuffle？？？
    ###
    xgb_regressor = train(df['2016-01': '2016-09'], df['2016-10'], params, verbose=verbose)
    # print('cross_validation...')

    test = df['2016-11': '2016-12']
    test_X = test.drop('lagging0', axis=1).values
    test_y = test['lagging0'].values

    y_pre = xgb_regressor.predict(test_X)
    print_error(test_y, y_pre)
    if save:
        file_path = './result.csv'
        if os.path.exists(file_path) == 0:
            with open(file_path, 'w') as fp:
                fp.write('n_estimators,learning_rate,max_depth,mape1,mape2,mape3,mape4,mean,MAPE\n')
                fp.close()
        with open('./result.csv', 'a') as fp:
            fp.write('%d,%2f,%d,' % (params['n_estimators'], params['learning_rate'], params['max_depth']))
            fp.write('%6f,%6f,%6f,%6f,%6f,%6f\n' % (train_mape[0], train_mape[1], train_mape[2], train_mape[3],
                                                    sum(train_mape)/len(train_mape), mape(test_y, y_pre)))
    print('train_mape: ', train_mape, 'mean_mape: ', np.mean(train_mape))
    print('test_mape', mape(test_y, y_pre))
    # 可视化特征重要程度
    if vis:
        train_feature = df.drop('lagging0', axis=1).columns.values
        feature_vis(xgb_regressor, train_feature)
    return 0


if __name__ == '__main__':
    df = pd.read_csv('./data/SR99_VDS1005210_2016_fill.csv',
                     delimiter=';', parse_dates=True, index_col='datetime')
    df = create_feature(df, lagging=5).dropna()
    # params_grid = {'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    #                'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    #                'max_depth': [3, 4, 5]}
    params_grid = {'learning_rate': [0.25], 'n_estimators': [1000], 'max_depth': [3]}
    params = ParameterGrid(params_grid)
    for param in params:
        print(param)
        cross_validation(df, param, save=False, vis=True, verbose=True)
