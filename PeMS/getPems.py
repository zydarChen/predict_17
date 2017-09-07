# -*- coding: utf-8 -*-
import datetime
import requests
import pandas as pd


def get_time_id(date):
    return 1362096000 + (date - datetime.datetime(2013, 3, 1)).days * 86400


def grt_url(formatter):
    cook = {
        'Cookie': '__utma=158387685.2041717991.1504705275.1504705275.1504705275.1; '
                  '__utmz=158387685.1504705275.1.1.utmcsr=google|utmccn=(organic)|'
                  'utmcmd=organic|utmctr=(not%20provided); '
                  'PHPSESSID=fbcd9ea7a9b773d536fb97d1eaa3e389; __utmt=1; '
                  '__utma=267661199.322005756.1504705334.1504774622.1504784751.7; __utmb=267661199.1.10.1504784751; '
                  '__utmc=267661199; __utmz=267661199.1504750317.3.3.utmcsr=google|utmccn=('
                  'organic)|utmcmd=organic|utmctr=(not%20provided)'}

    url = 'http://pems.dot.ca.gov/?report_form=1&dnode=Freeway&content=spatial&' \
          'tab=contours&export=xlsx&fwy={0[fwy]}&dir={0[dir]}&' \
          's_time_id={0[s_time_id]:d}&s_time_id_f={0[m]:02d}%2F{0[d]:02d}%2F{0[y]}&' \
          'from_hh=0&to_hh=23&start_pm=0&end_pm=2.49&lanes=&station_type=ml&' \
          'q=flow&colormap=30%2C31%2C32&sc=auto&ymin=&ymax=&view_d=2 '\
        .format(formatter)
    html = requests.get(url, cookies=cook)
    save_name = 'data/SR{0[fwy]}-{0[dir]}{0[y]}{0[m]:02d}{0[d]:02d}.xlsx'.format(formatter)
    print '开始保存' + save_name
    with open(save_name, 'wb') as f:
        f.write(html.content)

if __name__ == '__main__':
    select_dir = {'fwy': 90, 'dir': 'W', 's_time_id': 1362096000, 'y': 2013, 'm': 3, 'd': 1}
    date_list = pd.date_range('20130101', '20130301')
    for date in date_list:
        select_dir['y'] = date.year
        select_dir['m'] = date.month
        select_dir['d'] = date.day
        select_dir['s_time_id'] = get_time_id(date)
        grt_url(select_dir)
