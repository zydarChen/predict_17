# -*- coding: utf-8 -*-

import pandas as pd
from time import time


def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=';', parse_dates=['date'])
    return 0


if __name__ == '__main__':
    global_start_time = time()
    epochs = 1
    seq_len = 96
    print('>>> loading data... ')
    df_data = load_data('data/HE_2013_M25LM326_6_10.csv')
