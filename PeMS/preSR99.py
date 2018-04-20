# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PeMS.utils import *


def fill_nan(df, data_fill_path):
    # 2013年3月13日 2:00-3:00数据缺失，用历史值进行填充
    date_range = pd.date_range('2016-01-01 00:00:00', '2017-01-05 23:59:00', freq='5min')
    # print(np.setxor1d(date_range.values, df['5 Minutes'].unique()))
    df = df.set_index('5 Minutes').reindex(date_range).reset_index()  # 缺失数据填充为NAN
    df = df.rename(columns={'index': '5 Minutes'})

    df['weekday'] = df['5 Minutes'].map(lambda x: x.weekday())
    df['time'] = df['5 Minutes'].map(lambda x: '%02d%02d' % (x.hour, x.minute))
    # 同一时间片，周一到周日均值
    df_day_mean = df.groupby(['weekday', 'time'])['Flow (Veh/5 Minutes)'].mean().reset_index()
    df_day_mean = df_day_mean.rename(columns={'Flow (Veh/5 Minutes)': 'flow@mean'})
    df = pd.merge(df, df_day_mean, on=['weekday', 'time'], how='left')
    # df_nan = df.loc[df['Flow (Veh/5 Minutes)'].isnull()]  # 缺失数据段
    # 使用'flow_mean'列对'Flow (Veh/5 Minutes)'列进行缺失值填充
    df['Flow (Veh/5 Minutes)'] = df['Flow (Veh/5 Minutes)'].fillna(df['flow@mean'])
    # save
    df = df.rename(columns={'5 Minutes': 'datetime', 'Flow (Veh/5 Minutes)': 'flow_5'})
    df[['datetime', 'flow_5']].to_csv(data_fill_path,
                                      header=True, index=None, sep=';', mode='w')
    # 画图
    # tmp = df.loc[(df['datetime'].dt.month == 3) & (df['datetime'].dt.day == 13), :]
    # tmp['flow_5'].plot()
    # df_nan['flow@mean'].plot()
    # plt.show()
    return 0


def rw(df, display=False):
    # Random Walk
    # 十、十一月为测试集
    expect = df['2016-10': '2016-11'].values.flatten()
    predict = df.shift(-1)['2016-10': '2016-11'].values.flatten()
    print('\nRandom Walk:')
    print_error(expect, predict)
    if display:
        plot_results(expect[1000:1288], predict[1000:1288])
    return 0


def ha(df, alpha1=0.2, alpha2=0.2):
    # Historical Average
    df['weekday'] = df.index.weekday
    df['time'] = df.index.map(lambda x: '%02d%02d' % (x.hour, x.minute))
    df_week_mean = df.groupby(['weekday', 'time'])['flow'].mean().reset_index()
    df_week_mean = df_week_mean.rename(columns={'flow': 'flow@mean'})
    df = pd.merge(df.reset_index(), df_week_mean, on=['weekday', 'time'], how='left')
    df = df.set_index('datetime')

    # Naive HA
    expect = df['2016-10': '2016-11']['flow'].values.flatten()
    predict_ha = df['2016-10': '2016-11']['flow@mean'].values.flatten()
    print('\nHistorical Average:')
    print_error(expect, predict_ha)

    # EXPRW & Deviation from HA
    df['flow@ha'] = 0
    df['flow@dha'] = 0
    df.loc[: '2016-01-07', 'flow@dha'] = df[: '2016-01-07']['flow@mean']
    df.loc[: '2016-01-07', 'flow@ha'] = df[: '2016-01-07']['flow@mean']  # 第一周数据用历史平均值填充
    flow_ha = df['flow@ha'].values  # 视图
    s = df['flow@dha'].copy().values  # 深拷贝
    flow_dha = df['flow@dha'].values  # 浅拷贝
    backout = sum(flow_ha != 0)  # 回到上周的同一时间点
    flow = df['flow'].values
    length = len(flow)

    for x in range(backout, length-1):
        flow_ha[x] = alpha1 * flow[x-backout] + (1-alpha1) * flow_ha[x-backout]  # EXPRW
        s[x] = alpha2 * flow[x] + (1 - alpha2) * s[x-backout]
        flow_dha[x+1] = flow[x] * s[x-backout + 1] / s[x]  # Deviation from HA

    predict_exprw = df['2016-10': '2016-11']['flow@ha'].values.flatten()
    predict_dha = df['2016-10':'2016-11']['flow@dha'].values.flatten()
    print('\nEXPRW:')
    print_error(expect, predict_exprw)
    print('\nDHA:')
    print_error(expect, predict_dha)
    return 0


if __name__ == '__main__':
    folder = './data/SR99'
    data_path = './data/SR99_VDS1005210_2016.csv'
    data_fill_path = './data/SR99_VDS1005210_2016_fill.csv'
    if not os.path.exists(data_fill_path):
        if not os.path.exists(data_path):
            print('>>> 开始合并excel')
            excel2df(folder, data_path)
        df = pd.read_csv(data_path, delimiter=';', parse_dates=['5 Minutes'])
        print('>>> 开始缺失值填充')
        fill_nan(df, data_fill_path)
    df = pd.read_csv(data_fill_path, delimiter=';', parse_dates=True, index_col='datetime')
    # df = df.resample('15min', closed='right', label='right').sum()
    # df = df.resample('30min', closed='right', label='right').sum()
    # df = df.resample('45min', closed='right', label='right').sum()
    # df = df.resample('60min', closed='right', label='right').sum()
    print(df.head())
    # rw(df, True)
    ha(df.rename(columns={'flow_5': 'flow'}), alpha1=0.15, alpha2=0.15)
