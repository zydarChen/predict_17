# -*- coding: utf-8 -*-
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


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
        df.to_csv(save_path, index=None)
    return 0


def mae(y_true, y_predict):
    # Mean Absolute Error
    assert len(y_true) == len(y_predict)
    return np.mean(np.abs(y_true - y_predict))


def mre(y_true, y_predict):
    # Mean Relative Error
    assert len(y_true) == len(y_predict)
    return np.mean(np.abs(y_true - y_predict)/y_true) * 100


def mape(y_true, y_predict):
    # Mean Absolute Percentage Error
    assert len(y_true) == len(y_predict)
    return np.mean(np.abs(y_true - y_predict)/np.abs(y_true)) * 100


def xgb_mape(y_predict, dtrain):
    y_true = dtrain.get_label()
    return 'mape', np.mean(np.abs(y_true - y_predict)/np.abs(y_true)) * 100


def xgb_mapeobj(y_true, y_pred):
    grad = np.sign(y_pred - y_true) / y_true
    hess = 1/(y_true ** 2)
    grad[(y_true == 0)] = 0
    hess[(y_true == 0)] = 0
    return grad, hess


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


def plot_results(true_data, predicted_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    # plt.savefig('../res/predict.svg', format='svg')
    plt.show()


def feature_vis(model, feature):
    length = len(feature)
    model_feature_values = model.feature_importances_
    assert len(model_feature_values) == length
    rank = np.argsort(-model_feature_values)  # np.argsort 从小到大的索引值
    rank_feature = [feature[i] for i in rank]
    plt.figure(figsize=(20, 10))
    plt.title('Feature Importance')
    plt.bar(range(length), model_feature_values[rank])
    plt.xticks(range(length), rank_feature, rotation=90)
    plt.savefig('../res/feature_vis.png', format='png')
    # plt.show()



