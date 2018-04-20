#! /usr/bin/env python2
# -*- coding: utf-8 -*-
# http://pems.dot.ca.gov/?report_form=1&dnode=Freeway&content=elv&export=xls&fwy=10&dir=E&_time_id=1524089002&_time_id_f=04%2F18%2F2018&eqpo=&tag=&st_cd=on&st_ch=on&st_ff=on&st_hv=on&st_ml=on&st_fr=on&st_or=on&start_pm=.17&end_pm=239.92
import json
from datetime import datetime, timedelta
import requests
from PeMS.const import const
from urllib import urlencode
import os
from pandas import date_range, read_csv
from tqdm import tqdm
import time
from bs4 import BeautifulSoup
from utils import excel2df
# import pandas as pd


def time2time_id(time='19930101'):
    """
    将时间字符串转换成PeMS的时间ID
    :param time: str
    :return: int
    """
    # 19930101 00:00 = 725846400
    base_time = datetime(1993, 1, 1)
    base_time_id = 725846400
    try:
        delta = datetime.strptime(time, '%Y%m%d') - base_time
    except ValueError:
        raise
    time_id = base_time_id + delta.days * 24 * 3600 + delta.seconds
    return time_id


def get_vds(fwy=99, direction='N', time='20180419', path='./data/vds', vis=False):
    """
    下载VDS信息
    :param fwy: int|str, 高速路编号
    :param direction: [ESWN], 道路方向
    :param time: str, 获取时间
    :param vis: bool, 是否打印信息
    :param path: str, 保存路径
    :return: .xlsx, freeway上所有VDS的信息
    """
    url = 'http://pems.dot.ca.gov'
    redirect_head = const.VDS_HEAD  # 从常量表中取出URL_HEAD
    time_id = time2time_id(time)
    time_f = '%s/%s/%s 00:00' % (time[:4], time[4:6], time[6:])
    redirect_dict = {
        'fwy': fwy,
        'dir': direction,
        '_time_id': time_id,
        '_time_id_f': time_f,
    }
    redirect = redirect_head + urlencode(redirect_dict)
    # 提交表单
    const.DATA_FROM['redirect'] = redirect
    html = requests.post(url, data=const.DATA_FROM)
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = path + '/{fwy}-{direction}.xlsx'.format(fwy=fwy, direction=direction)
    if vis:
        print('>>> 开始保存 ' + save_name)
    with open(save_name, 'wb') as f:
        f.write(html.content)
    return 0


def save_all_vds(fwy_path='./data/freeway_name.json', path='./data/vds'):
    """
    下载全路段VDS
    :param fwy_path: freewy_name路径
    :param path: 保存路径
    :return:
    """
    with open(fwy_path, 'r') as fp:
        data = json.load(fp)
    for key in tqdm(data.keys()):
        fwy = data[key][0]
        direction = data[key][1]
        save_name = path + '/{fwy}-{direction}.xlsx'.format(fwy=fwy, direction=direction)
        # 如果文件已存在，则跳过
        if os.path.exists(save_name):
            continue
        get_vds(fwy=fwy, direction=direction, path=path)
        # 如果文件大小小于2K，说明没有保存成功
        while True:
            if os.path.getsize(save_name) > 2000:
                break
            time.sleep(5)
            get_vds(fwy=fwy, direction=direction, path=path)


def get_detector_info(fwy=99, direction='N', station_id=1005210):
    """
    获取单个检测器信息，包括路宽，经纬度等
    :param fwy:
    :param direction:
    :param station_id:
    :return:
    """
    url = 'http://pems.dot.ca.gov'
    redirect_head = '/?dnode=VDS&'
    redirect_dict = {
        'fwy': fwy,
        'dir': direction,
        'station_id': station_id,
    }
    redirect = redirect_head + urlencode(redirect_dict)
    # 提交表单
    const.DATA_FROM['redirect'] = redirect
    html = requests.post(url, data=const.DATA_FROM)
    soup = BeautifulSoup(html.content, 'html.parser')
    table = soup.find_all('table')[:2]
    fwy_info = []
    for td in table[0].find_all('td'):  # Roadway Information (from TSN)表格
        fwy_info.append(td.text.replace(',', ';'))  # 可能存在，分隔符
    fwy_info = fwy_info[1:]
    last_log = table[1].find_all('tr')[-1].text.strip().split('\n')  # Change Log 取最后一行
    return fwy_info, last_log


def merge_vds(path='./data/vds', detector_path='./data/all_detector.csv', save_path='./data/all_detector_new.csv', fwy_path='./data/freeway_name.json'):
    """
    下载全部检测站信息，包括经纬度，路宽等
    :param path: freeway检测站文件路径
    :param detector_path: all_detector.csv路径
    :param save_path: 保存路径
    :param fwy_path: freeway_name.json路径
    :return:
    """
    if not os.path.exists(detector_path):
        print('>>> 开始合并excel')
        excel2df(path, detector_path)
    with open(fwy_path, 'r') as fp:
        data = json.load(fp)
    # 一行一行读，一行一行写
    with open(save_path, 'w') as fp:
        head = 'fwy_name,district,county,city,ca_pm,abs_pm,length,station_id,name,lanes,type,sensor_type,hov,ms_id,' \
               'irm,road_width,lane_width,inner_shoulder_width,inner_shoulder_treated_width,outer_shoulder_width,' \
               'outer_shoulder_treated_width,design_speed_limit,functional_class,inner_median_type,' \
               'inner_median_width,terrain,population,barrier,surface,roadway_use,change_last_date,status,lat,lng,' \
               'fwy,direction '
        fp.write(head + '\n')
    cnt = 0
    with open(detector_path, 'r') as fp:
        for line in tqdm(fp):
            if cnt == 0:  # head不处理
                cnt += 1
                continue
            line = line.strip()
            fwy_name = line.split(',')[0]
            station_id = line.split(',')[7]
            fwy = data[fwy_name][0]
            direction = data[fwy_name][1]
            # print(fwy_name, fwy, direction, station_id)
            try:
                info = get_detector_info(fwy=fwy, direction=direction, station_id=int(station_id))
                # Roadway Information (from TSN)表格
                add_line1 = ','.join([s for i, s in enumerate(info[0]) if i % 2 == 1])
                add_line2 = info[1][0] + ',' + info[1][1] + ',' + info[1][7] + ',' + info[1][8]  # Change Log
                add_line3 = fwy + ',' + direction
                with open(save_path, 'a') as fw:
                    fw.write(line + ',' + add_line1 + ',' + add_line2 + ',' + add_line3 + '\n')
            except:
                # with open(path + '/re_down_new.csv', 'a') as fw1:
                #     fw1.write(fwy + ',' + direction + ',' + station_id + '\n')
                print('[Error] Something is wrong with fwy=%s direction=%s station_id=%s' % (fwy, direction, station_id))
            cnt += 1
            # if cnt % 1000 == 0:
            #     print('>>> %d/18101' % cnt)
    return 0


def re_merge_vds():
    """
    弃用，新函数re_download_vds
    由于VPN不稳定，可能部分数据没有下载，重新下载错误数据
    :return:
    """
    df = read_csv('./data/all_detector.csv')
    with open('./data/re_down.csv', 'r') as fp:
        lines = fp.readlines()
    for line in tqdm(lines):
        lst = line.strip().split(',')
        fwy = lst[0]
        direction = lst[1]
        station_id = lst[2]
        try:
            info = get_detector_info(fwy=int(fwy), direction=direction, station_id=int(station_id))
            # Roadway Information (from TSN)表格
            line = df.loc[df.ID == int(station_id)].values[0]
            line_str = ','.join(map(str, line)).replace('nan', '')
            line1 = ','.join([s for i, s in enumerate(info[0]) if i % 2 == 1])
            line2 = info[1][0] + ',' + info[1][1] + ',' + info[1][7] + ',' + info[1][8]  # Change Log
            line3 = fwy + ',' + direction
            with open('./data/error.csv', 'a') as fw:
                fw.write(line_str + ',' + line1 + ',' + line2 + ',' + line3 + '\n')
        except:
            with open('./data/re_down_new.csv', 'a') as fw1:
                fw1.write(fwy + ',' + direction + ',' + station_id + '\n')
            print('[Error] Something is wrong with fwy=%s direction=%s station_id=%s' % (fwy, direction, station_id))
    return 0


def re_download_vds(detector_list='./data/detector_list.json', all_detector='./data/all_detector.csv', exists_detector='./data/all_detector_merge_info.csv'):
    """
    由于VPN不稳定，可能部分数据没有下载，重新下载错误数据
    :param detector_list:
    :param all_detector:
    :param exists_detector:
    :return:
    """
    with open(detector_list, 'r') as fp:
        detector_dict = json.load(fp)
    df = read_csv(exists_detector)
    exists_detector_list = map(str, df['station_id'].tolist())  # 取出已存在的id
    df = read_csv(all_detector)
    all_detector_list = map(str, detector_dict.keys())
    diff = list(set(all_detector_list).difference(set(exists_detector_list)))  # 取差集
    while diff:
        station_id = diff.pop(0)
        fwy = detector_dict[station_id][0]
        direction = detector_dict[station_id][1]
        try:
            info = get_detector_info(fwy=int(fwy), direction=direction, station_id=int(station_id))  # 爬取信息
            # Roadway Information (from TSN)表格
            line = df.loc[df.ID == int(station_id)].values[0]
            line_str = ','.join(map(str, line)).replace('nan', '')
            line1 = ','.join([s for i, s in enumerate(info[0]) if i % 2 == 1])
            line2 = info[1][0] + ',' + info[1][1] + ',' + info[1][7] + ',' + info[1][8]  # Change Log
            line3 = fwy + ',' + direction
            with open(exists_detector, 'a') as fw:
                fw.write(line_str + ',' + line1 + ',' + line2 + ',' + line3 + '\n')  # 追加写入文件
        except:
            print('[Error] Something is wrong with fwy=%s direction=%s station_id=%s' % (fwy, direction, station_id))
            diff.append(station_id)  # 出问题ID需要重新处理
            print('Now is %s, number of station is %d' % (time.strftime('%H:%M:%S', time.localtime()), len(diff)))
            time.sleep(300)
    return 0


def get_data(start='20180420', end='20180420', station_id=1005210, q='flow', q2='speed', gn='5min',
             path='./data', fwy_name='SR99-N', vis=False):
    """
    下载单个检测站在[start, end]间的数据
    :param vis:
    :param fwy_name:
    :param path:
    :param start: str
    :param end: str
    :param station_id: int
    :param q: str, flow|speed|occ|truck_flow|truck_prop|truck_vmt|truck_vht|q|tti|vmt|vht
    :param q2: str
    :param gn: 5min|hour|day|[week|month]
    :return: 保存数据
    """
    flag = True
    time_delta = datetime.strptime(end, '%Y%m%d') - datetime.strptime(start, '%Y%m%d')
    if time_delta.days < 0:
        flag = False
        print('>>> start_time is not less than end_time')
    elif gn == '5min' and time_delta.days >= 7:
        flag = False
        print('Max Range of 5min: 1 week')
    elif gn == 'hour' and time_delta.days >= 90:
        flag = False
        print('Max Range of hour: 3 months')
    if flag:
        redirect_head = const.DATA_HEAD  # 从常量表中取出HEAD
        url = 'http://pems.dot.ca.gov'
        start_id = time2time_id(start)
        end_id = time2time_id(end) + 86340  # 转成23:59
        start_id_f = '%s/%s/%s 00:00' % (start[:4], start[4:6], start[6:])
        end_id_f = '%s/%s/%s 23:59' % (end[:4], end[4:6], end[6:])
        redirect_dict = {
            'station_id': station_id,
            's_time_id': start_id,
            's_time_id_f': start_id_f,
            'e_time_id': end_id,
            'e_time_id_f': end_id_f,
            'q': q,
            'q2': q2,
            'gn': gn
        }
        redirect = redirect_head + '&' + urlencode(redirect_dict)
        # 提交表单
        data = {
            'redirect': redirect,
            'username': 'zydarchen@outlook.com',
            'password': 'treep9:rQ',
            'login': 'Login',
        }
        html = requests.post(url, data=data)
        save_path = path + '/' + fwy_name + '/' + str(station_id)
        if not os.path.exists(save_path):  # ./data/SR99-N/1005210文件夹是否存在
            os.makedirs(save_path)
        save_name = save_path + '/VDS{station_id}-{start}-{end}.xlsx'.format(station_id=station_id, start=start, end=end)
        if vis:
            print('>>> 开始保存 ' + save_name)
        with open(save_name, 'wb') as f:
            f.write(html.content)
    return 0


def download_data(detector_list='./data/detector_list.json', path='./data', q='flow', q2='speed', gn='5min'):
    """
    下载全部数据
    :param detector_list:
    :param path:
    :param q:
    :param q2:
    :param gn:
    :return:
    """
    with open(detector_list, 'r') as fp:
        detector_dict = json.load(fp)
    for station_id in tqdm(detector_dict):
        date_list = date_range('20160101', '20161231', freq='7D').tolist()
        while date_list:
            start = date_list.pop(0)
            end = start + timedelta(days=6)
            start_str = start.strftime('%Y%m%d')
            end_str = end.strftime('%Y%m%d')
            fwy_name = '-'.join(detector_dict[station_id])
            try:
                get_data(start=start_str, end=end_str, station_id=int(station_id), path=path, fwy_name=fwy_name,
                         q=q, q2=q2, gn=gn, vis=False)
            except:
                print('[Error] Something is wrong with station_id=%s start=%s end=%s' % (station_id, start_str, end_str))
                date_list.append(start)
                print('Now is %s, number of date_list is %d' % (time.strftime('%H:%M:%S', time.localtime()), len(date_list)))
                time.sleep(300)


if __name__ == '__main__':
    # save_all_vds()  # 下载全部freeway的VDS
    # print(get_detector_info(1, 'N', 1221170))
    # merge_vds()
    # get_data(start='20180413', end='20180419', station_id=1005210, q='flow', q2='speed', gn='5min',
    #          path='./data', fwy_name='SR99-N', vis=False)
    # re_download_vds()
    download_data()
    pass
