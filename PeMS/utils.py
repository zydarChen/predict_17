# -*- coding: utf-8 -*-
import os
import pandas as pd
from tqdm import tqdm
import numpy as np


def excel2df(path, save_path):
    """将path下的所有excel文件合并成DataFrame，所有excel具有相同的表头
    Args:
        path: string, 文件夹路径
        save_path: string, 保存路径
    Returns:
    """
    for parent, _, names in os.walk(path):
        # 获取表头
        columns = pd.read_excel(os.path.join(parent, names[0])).columns
        df = pd.DataFrame(columns=columns)
        for name in tqdm(names):
            df_next = pd.read_excel(os.path.join(parent, name))
            df = pd.concat([df, df_next], axis=0, ignore_index=True)
        df.to_csv(save_path, header=True, index=None, sep=';', mode='w')
    return 0


def mae(y_true, y_predict):
    # Mean Absolute Error
    assert len(y_true) == len(y_predict)
    return np.mean(np.abs(y_true - y_predict))


def mape(y_true, y_predict):
    # Mean Relative Error
    assert len(y_true) == len(y_predict)
    return np.mean(np.abs(y_true - y_predict)/y_true) * 100


def mse(y_true, y_predict):
    # Mean Square Error
    assert len(y_true) == len(y_predict)
    return np.square(np.mean((y_true - y_predict) ** 2))


def rmse(y_true, y_predict):
    # Root Mean Square Error
    assert len(y_true) == len(y_predict)
    return np.mean((y_true - y_predict) ** 2)


def print_error(y_true, y_predict):
    print('MSE: %.3f' % mse(y_true, y_predict))
    print('RMSE: %.3f' % rmse(y_true, y_predict))
    print('MAE: %.3f' % mae(y_true, y_predict))
    print('MAPE: %.3f%%' % mape(y_true, y_predict))


