# -*- coding: utf-8 -*-
import re
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
