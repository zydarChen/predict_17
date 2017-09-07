# -*- coding: utf-8 -*-
import urllib
import hashlib
import json
import requests
import os
from lxml import etree

# 根据经纬度获取地址信息
def geocoding(lng_lat):
    """根据经纬度获取地址信息
        :param lng_lat: str, 经纬度如'31.1219830, 121.3743330'
        :return: dict, 返回地址信息字典，包括省、市、区县、街道
    """
    add = {}
    ak = 'WtPdP20GRKi9Ord9o8bGmfMYRYK3pgMZ'
    sk = 'G0GUQW7lx7bQBzw2wcGnPTVq8kCQkbNu'
    queryStr = '/geocoder/v2/?location=' + lng_lat + '&output=json&ak=' + ak

    # 对queryStr进行转码，safe内的保留字符不转换
    encodedStr = urllib.quote(queryStr, safe="/:=&?#+!$,;'@()*[]")
    # 在最后直接追加上yoursk
    rawStr = encodedStr + sk
    sn = hashlib.md5(urllib.quote_plus(rawStr)).hexdigest()
    url = 'http://api.map.baidu.com' + queryStr + '&sn=' + sn  # 最终提交的url
    info_json = json.loads(urllib.urlopen(url).read())  # 解析返回的json格式数据

    add['province'] = info_json['result']['addressComponent']['province']  # 省
    add['city'] = info_json['result']['addressComponent']['city']  # 市
    add['district'] = info_json['result']['addressComponent']['district']  # 区县
    add['street'] = info_json['result']['addressComponent']['street']  # 街

    return add

# 根据城市区县拼音获取天气
def get_weather(address, ymd):
    """根据城市区县拼音获取天气
        :param address: str, 区县拼音，beijijng
        :param ymd: str, 日期
        :return: dict, 天气字典，包括白天天气，夜间天气，最高气温，最低气温，风向风速
    """
    wea = {}
    url = 'http://' + address + '.tianqi.com/' + ymd + '.html'
    html = requests.get(url).text
    wea['day'] = etree.HTML(html).xpath('//*[@id="today"]/ul/li[2]')[1].text  # 白天天气
    wea['night'] = etree.HTML(html).xpath('//*[@id="today"]/ul/li[2]')[2].text  # 夜间天气
    wea['temp_high'] = etree.HTML(html).xpath('//*[@id="t_temp"]/font[1]')[0].text  # 最高气温
    wea['temp_low'] = etree.HTML(html).xpath('//*[@id="t_temp"]/font[2]')[0].text  # 最低气温
    wea['wind'] = etree.HTML(html).xpath('//*[@id="today"]/ul/li[5]')[0].text  # 风向风速
    return wea

# 根据区县中文对应网站拼音格式
def get_city_dict():
    """根据区县中文对应网站拼音格式
        :return: dict, 返回区县与区县拼音对应关系字典
    """
    if os.path.exists('city_dict.txt'):
        with open('city_dict.txt', 'r') as fp:
            city_dict = json.load(fp)
    else:
        city_dict = {}
        url = 'http://lishi.tianqi.com'
        html = requests.get(url).text
        chars = etree.HTML(html).xpath('//*[@id="tool_site"]/div[2]/ul')  # 每个字母
        for char in chars:
            citys = char.xpath('li/a/text()')  # 提取区县名
            pinyins = char.xpath('li/a/@href')  # 提取区县链接
            num = len(citys)
            assert len(pinyins) == num
            for i in range(1,num):
                city_dict[citys[i]] = pinyins[i].split('/')[-2]  # 从区县链接中提取拼音
        with open('city_dict.txt', 'w') as fp:
            fp.write(json.dumps(city_dict))
    return city_dict


# 获取天气情况
def get_wea_fin(lng=23.152, lat=113.346, date='20141207'):
    """根据经纬度获取天气情况
        :param lng: num, 纬度
        :param lat: num, 经度
        :param date: str, 日期
        :return: dict, 日期，地址信息字典，天气信息字典
    """
    lng_lat = str(lng) + ',' + str(lat)
    address = geocoding(lng_lat)
    # province = address['province']  # 通过百度API提取的省名称
    city = address['city']  # 市
    district = address['district']  # 区县
    weather = {}
    if district in citys:
        weather = get_weather(city_dict[district], str(date))
    elif district[:-1] in citys:  # 天气网的部分区县没有以区县结尾，如，闵行区表示为闵行
        weather = get_weather(city_dict[district[:-1]], str(date))
    elif district == '' and city != '':  # 不存在区县信息，取城市信息
        weather = get_weather(city_dict[city], str(date))
    else:
        print '对不起，无法查到您给的经纬度信息，请输入中国大陆范围内经纬度信息。'.decode('utf-8').encode('GB2312')
    return str(date), address, weather


city_dict = get_city_dict()
citys = city_dict.keys()

if __name__ == '__main__':
    lng_lat = raw_input('请输入经纬度信息，如23.152 113.346:'.decode('utf-8').encode('GB2312'))
    date = raw_input('请输入查询日期，如20141207:'.decode('utf-8').encode('GB2312'))
    if lng_lat and date:
        lng = lng_lat.split()[0]
        lat = lng_lat.split()[1]
        weather = get_wea_fin(lng, lat, date)
    else:
        weather = get_wea_fin()
    print '日期：'.decode('utf-8').encode('GB2312'), weather[0]
    print '地址信息: '.decode('utf-8').encode('GB2312'), weather[1]['province'], weather[1]['city'], weather[1]['district'], weather[1]['street']
    print '天气情况: '.decode('utf-8').encode('GB2312')
    print '白天天气: '.decode('utf-8').encode('GB2312'), weather[2]['day']
    print '夜间天气: '.decode('utf-8').encode('GB2312'), weather[2]['night']
    print '最高气温: '.decode('utf-8').encode('GB2312'), weather[2]['temp_high']
    print '最低气温: '.decode('utf-8').encode('GB2312'), weather[2]['temp_low']
    print '风速风向: '.decode('utf-8').encode('GB2312'), weather[2]['wind']
    raw_input('press any key to exit:')
