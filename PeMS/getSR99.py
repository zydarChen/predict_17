# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import requests
from PeMS.const import const
from urllib import urlencode
from pandas import date_range
from tqdm import tqdm
import time


def get_time_id(datetime1, datetime2):
    # 19930101 00:00 = 725846400
    datetime_base = datetime(1993, 1, 1)
    time_delta = datetime2 - datetime1
    if time_delta.days < 0:
        print('datetime1 is not less than datetime2')
    elif time_delta.days >= 7:
        print('Max Range: 1 week')
    else:
        time_delta1 = datetime1 - datetime_base
        datetime1_id = 725846400 + time_delta1.days*24*3600 + time_delta1.seconds
        datetime2_id = datetime1_id + time_delta.days * 24 * 3600 + time_delta.seconds
        # quote会默认忽略'/',quote_plus将' '编码为'+'
        # datetime1_f = quote_plus(datetime1.strftime('%m/%d/%Y %H:%M'), safe='')
        # datetime2_f = quote_plus(datetime2.strftime('%m/%d/%Y %H:%M'), safe='')
        return datetime1_id, datetime2_id


def get_data(dt1, dt2, station_id=1005210):
    redirect_head = const.DATA_HEAD  # 从常量表中取出HEAD
    url = 'http://pems.dot.ca.gov'
    dt = get_time_id(dt1, dt2)
    if dt:
        dt1_id, dt2_id = dt[0], dt[1]
        dt1_f, dt2_f = dt1.strftime('%m/%d/%Y %H:%M'), dt2.strftime('%m/%d/%Y %H:%M')
        redirect_dict = {
            'station_id': station_id,
            's_time_id': dt1_id,
            's_time_id_f': dt1_f,
            'e_time_id': dt2_id,
            'e_time_id_f': dt2_f,
            'q': 'flow',
            'q2': '',
            'gn': '5min'
        }
        redirect = redirect_head + '&' + urlencode(redirect_dict)
        print(redirect)
        # 提交表单
        data = {
            'redirect': redirect,
            'username': 'zydarchen@outlook.com',
            'password': 'treep9:rQ',
            'login': 'Login',
        }
        html = requests.post(url, data=data)
        save_name = 'data/{fwy}/VDS{station_id}-{dt1}-{dt2}.xlsx'.format(
            fwy='SR99', station_id=station_id, dt1=dt1.strftime('%Y%m%d'), dt2=dt2.strftime('%Y%m%d'))

        print('>>> 开始保存 ' + save_name[10:])
        with open(save_name, 'wb') as f:
            f.write(html.content)
    return 0


if __name__ == '__main__':
    # test
    # dt1 = datetime(2016, 3, 11)
    # dt2 = datetime(2016, 3, 17, 23, 59)
    # get_data(dt1, dt2, station_id=1005210)
    # 下载2016年数据
    date_list = date_range('20160101', '20161231', freq='7D')
    station_id = 1005210
    global_start_time = time.time()
    for date in tqdm(date_list):
        dt2 = date + timedelta(days=7) - timedelta(minutes=1)
        get_data(date, dt2, station_id=station_id)
    print('总用时 (s): ', time.time() - global_start_time)

