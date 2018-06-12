# -*- coding: utf-8 -*-
from math import radians, cos, acos
import pandas as pd
from datetime import datetime, timedelta


def haversine(lon1, lat1, lon2, lat2):
    """
    :param lon1: float, 经度1
    :param lat1: float, 纬度1
    :param lon2: float, 经度2
    :param lat2: float, 纬度2
    :return: float, 地球上两点之间的距离
    """
    def hav(theta):
        return (1 - cos(theta)) / 2.0
    # 转成弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # 计算haversine
    hav_h = hav(abs(lon2 - lon1)) + cos(lon1) * cos(lon2) * hav(abs(lat2 - lat1))
    h = acos(1 - 2*hav_h)
    r = 6371  # 地球半径，6371km
    return h * r


def get_weather_station(log=-121.13333, lat=37.746753):
    """
    :param log: float, PeMS检测站经度
    :param lat: float, PeMS检测站纬度
    :return: int, 最近邻的weather检测站
    """
    df_station = pd.read_csv('../data/df_station.csv', dtype=object)
    df_station['distance'] = df_station.longitude + '_' + df_station.latitude
    df_station['distance'] = df_station['distance'].map(
        lambda x: haversine(float(x.split('_')[0]), float(x.split('_')[1]), log, lat))
    return df_station.loc[df_station.distance.idxmin()]['station_id']


def str2float(str):
    try:
        num = float(str)
    except ValueError:
        try:
            num = float(str[:-1])
        except ValueError:
            num = float('nan')
    return num


def sky2str(sky):
    try:
        tmp = sky.replace('s', '').split(' ')  # 删除字符串中的's'
    except AttributeError:  # for nan
        return float('nan')
    if '*' in tmp:  # for '*'
        return float('nan')
    elif tmp[-1].isdigit():
        # 最后一位为数字，表示云层高度
        # 取倒数第二位
        # CLR:00
        try:
            res = tmp[-2]
        except IndexError:  # for '3'
            return float('nan')
    else:
        res = tmp[-1]
    return res


def weather2code(weather, code):
    i = {'AU': 0, 'AW': 1, 'MW': 2}.get(code)
    try:
        tmp = weather.replace('s', '').split('|')[i]  # 删除字符串中的's'
    except AttributeError:  # for nan
        return float('nan')
    tmp = tmp.replace('*', '')
    if not tmp:
        return float('nan')
    return tmp.split()


def pre_date(date):
    timestamp = round((date.hour * 60 + date.minute) / 5) * 5
    hour = timestamp // 60
    minute = timestamp - hour * 60
    if hour == 24:
        date = date + timedelta(1)
        return datetime(date.year, date.month, date.day, 23, minute, 0)
    return datetime(date.year, date.month, date.day, hour, minute, 0)


def weather_dataframe(log=-121.13333, lat=37.746753):
    station_id = get_weather_station(log, lat)
    df = pd.read_csv('../data/all_weather_new.csv', dtype=object, parse_dates=['date'])
    df = df.loc[(df.station_id == station_id) & (df.date.dt.year == 2016)].reset_index(drop=True)
    # 温度
    df['temp'] = df['temp'].apply(str2float).fillna(method='ffill')
    # 能见度
    df['visibility'] = df['visibility'].apply(str2float).fillna(method='ffill')
    # 风速
    df['wind_speed'] = df['wind_speed'].apply(str2float).fillna(method='ffill')
    # 风向
    df['wind_dir'] = df['wind_dir'].apply(str2float).fillna(method='ffill')
    # 降雨量
    df['precip'] = df['precip'].apply(str2float)  # 缺失太多，不直接填充
    # 云层情况
    # 'CLR:00', 'FEW:02', 'SCT:04', 'BKN:07', 'OVC:08', 'VV:09', 'X:10'
    df['sky'] = df['sky'].apply(sky2str).fillna(method='ffill')
    # 天气类型
    df['AU'] = df['weather'].apply(weather2code, code='AU')
    df['AW'] = df['weather'].apply(weather2code, code='AW')
    df['MW'] = df['weather'].apply(weather2code, code='MW')
    del df['weather']
    # date
    df['datetime'] = df['date'].apply(pre_date)
    return df.drop_duplicates('datetime')


if __name__ == '__main__':
    df_weather = weather_dataframe(log=-121.13333, lat=37.746753)
    print(df_weather.head())






