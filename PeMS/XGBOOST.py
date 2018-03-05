# -*- coding: utf-8 -*-
from __future__ import print_function
from PeMS.utils import *
import xgboost as xgb

# 数据平滑


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
    df_lagging['month'] = df.index.map(lambda date: date.month)
    # vacation
    # one-hot编码
    return pd.get_dummies(df_lagging, columns=['day_of_week', 'hour', 'minute'])


df = pd.read_csv('./data/SR99_VDS1005210_2016_fill.csv',
                 delimiter=';', parse_dates=True, index_col='datetime')
df = create_feature(df, 5).dropna()
train = df[: '2016-09']  # 1-9训练
test = df['2016-10':]  # 10、11测试
X_train = train.drop('lagging0', axis=1).values
X_test = test.drop('lagging0', axis=1).values
y_train = train['lagging0'].values
y_test = test['lagging0'].values
xgb = xgb.XGBRegressor(n_jobs=-1)
xgb.fit(X_train, y_train)
y_pre = xgb.predict(X_test)
print(mape(y_test, y_pre))
