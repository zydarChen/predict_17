# -*- coding: utf-8 -*-
import re
from zipfile import BadZipfile

from PeMS.download import get_detector_info
from tqdm import tqdm
from collections import defaultdict
import json
import pandas as pd


# data = defaultdict(list)
# with open('./PeMS/data/freeway_name.txt', 'r') as fp:
#     for line in fp:
#         # print line
#         # <option label="I10-E" value="/?dnode=Freeway&amp;fwy=10&amp;dir=E">I10-E</option>
#         name = re.search(r'[A-Z]{1,2}\d{1,3}[A-Z]?-[ESWN]', line).group(0)  # 公路名
#         data[name].append(re.search(r'fwy=\d+', line).group(0)[4:])  # fwy
#         data[name].append(re.search(r'dir=[ESWN]', line).group(0)[4:])  # dir


# with open('./PeMS/data/freeway_name.json', 'w') as fp:
#     json.dump(data, fp)

# with open('./PeMS/data/freeway_name.json', 'r') as fp:
#     data = json.load(fp)
# data = defaultdict(list)
# df = pd.read_csv('./PeMS/data/all_detector.csv')
# line = df.loc[df.ID == 1221176].values[0]
# line_str = ','.join(map(str, line)).replace('nan', '')
# print(line_str)
# with open('./PeMS/data/error.csv', 'w') as fp:
#     head = 'fwy_name,district,county,city,ca_pm,abs_pm,length,station_id,name,lanes,type,sensor_type,hov,ms_id,' \
#            'irm,road_width,lane_width,inner_shoulder_width,inner_shoulder_treated_width,outer_shoulder_width,' \
#            'outer_shoulder_treated_width,design_speed_limit,functional_class,inner_median_type,' \
#            'inner_median_width,terrain,population,barrier,surface,roadway_use,change_last_date,status,lat,lng,' \
#            'fwy,direction '
#     fp.write(head + '\n')
# with open('./PeMS/data/error.txt', 'r') as fp:
#     for line in tqdm(fp):
#         # print line
#         # 1090it [55:13,  3.04s/it][Error] Something is wrong with fwy=10 direction=W station_id=816381
#         fwy = re.search(r'fwy=\d+', line).group(0)[4:]  # fwy
#         direction = re.search(r'direction=[ESWN]', line).group(0)[-1]  # direction
#         station_id = re.search(r'station_id=\d+', line).group(0)[11:]  # station_id
#         try:
#             info = get_detector_info(fwy=int(fwy), direction=direction, station_id=int(station_id))
#             # Roadway Information (from TSN)表格
#             line = df.loc[df.ID == int(station_id)].values[0]
#             line_str = ','.join(map(str, line)).replace('nan', '')
#             line1 = ','.join([s for i, s in enumerate(info[0]) if i % 2 == 1])
#             line2 = info[1][0] + ',' + info[1][1] + ',' + info[1][7] + ',' + info[1][8]  # Change Log
#             line3 = fwy + ',' + direction
#             with open('./PeMS/data/error.csv', 'a') as fw:
#                 fw.write(line_str + ',' + line1 + ',' + line2 + ',' + line3 + '\n')
#         except:
#             with open('./PeMS/data/error_new.csv', 'a') as fw1:
#                 fw1.write(fwy + ',' + direction + ',' + station_id + '\n')
#             print('[Error] Something is wrong with fwy=%s direction=%s station_id=%s' % (fwy, direction, station_id))
import requests

# def login(username='zydarchen@outlook.com', password='treep9:rQ'):
#     session = requests.Session()
#     form_data = {
#         'redirect': '',
#         'username': username,
#         'password': password,
#         'login': 'Login',
#         }
#     sess = session.post('http://pems.dot.ca.gov', data=form_data)
#     for _ in range(100):
#         first = session.get('http://pems.dot.ca.gov')
#         print(first.status_code == 200)
#
#
# login()
import os
# for parent, _, file_names in os.walk('./PeMS/data/flow_data/101-S'):
#     for file_name in file_names:
#         cur_file = os.path.join(parent, file_name)
#         if os.path.getsize(cur_file) < 1024:
#             print os.path.getsize(cur_file)
# #             print cur_file
# for parent, _, file_names in os.walk('./PeMS/data/flow_data'):
#     for file_name in file_names:
#         cur_file = os.path.join(parent, file_name)
#         # print(cur_file)
#
#         continue
import xlrd
import re
from urllib import unquote_plus
from datetime import datetime, timedelta


def time_id2time(time_id=725846400):
    """
    将PeMS的时间ID转成时间字符串
    :param time_id: int
    :return: str
    """
    # 19930101 00:00 = 725846400
    base_time_id = 725846400
    base_time = datetime(1993, 1, 1)
    delta = timedelta(seconds=time_id - base_time_id)
    time_new = base_time + delta
    return datetime.strftime(time_new, '%Y%m%d')


def check(path='.\\data\\flow_data\\87-N\\405569\\VDS405569-20160909-20160915.xlsx'):
    """
    检查下载的文件与内容是否对应
    :param path:
    :return: 没出错返回False，出错返回错误的(fwy, station_id_path, start_path, end)
    BUG,需要匹配s_time_id，而不是s_time_id_f
    """
    table = xlrd.open_workbook(path).sheet_by_index(1)
    url = table.row_values(2)[2]
    station_id = re.search(r'station_id=(\d+)&?', url).groups()[0]
    start = re.search(r's_time_id_f=(.{22})&?', url).groups()[0]
    start = unquote_plus(start)[:10].replace('/', '')
    station_id_path = path.split('\\')[-2]
    start_path = path.split('-')[-2]
    if station_id == station_id_path and start == start_path:
        return []
    fwy = path.split('\\')[-3]
    end = path.split('-')[-1][:-5]
    return fwy, station_id_path, start_path, end


def check_all(path='.\\data\\flow_data'):
    for parent, _, file_names in os.walk(path):
        for file_name in file_names:
            cur_file = os.path.join(parent, file_name)
            try:
                lst = check(cur_file)
                if lst:
                    print(lst)
            except BadZipfile:
                print('BadZipfile')
            except:
                print('[Error] Something is wrong with %s' % cur_file)


print(check(path='.\\PeMS\\data\\flow_data\\1-N\\500010152\\VDS500010152-20160101-20160107.xlsx'))
