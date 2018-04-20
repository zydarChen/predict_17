# -*- coding: utf-8 -*-
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

# 载入数据
df = pd.read_csv('./data/SR99_VDS1005210_2016_fill.csv', delimiter=';', parse_dates=True, index_col='datetime')

# 8、9、10月为训练数据
train = df['2016-08': '2016-10'].values[: 100]
test = df['2016-11': '2016-12'].values[: 100]
history = [x for x in train]
predictions = []
test_num = len(test)
for t in range(test_num):
    model = ARIMA(history, order=(1, 0, 1))
    model_fit = model.fit(disp=0)
    model_fit.fittedvalues()  # 返回真实值减去残差
    model_fit.predict()
    # yhat = model_fit.forecast()[0]
    # predictions.append(yhat)
    # history.append(test[t])
