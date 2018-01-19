# ARIMA学习笔记

## 基础知识
- 严平稳：序列所有的统计性质不会随着时间的推移而发生变化。
- 宽平稳：常数均值；自协方差函数和自相关系数只依赖于时间的平移长度而与时间的起止点无关。
- 平稳序列通常具有短期相关性。该性质用自相关系数来描述就是随着延迟期数k的增加，平稳序列的自相关系数会很快衰减向零。
- 如何判定是否平稳：在自相关图上，随着延迟期数k的增加,自相关系数很快衰减向零，且之后始终控制在2倍的标准差范围内。
- 自相关系数：平稳MA(q)模型的自相关系数q阶截尾。
- 偏自相关系数：单纯测度x_{t+k}对x_{t}的影响，而排除中间k-1个随机变量对x_{t}的影响。平稳AR(p)模型的偏自相关系数具有p步截尾性（p步之后为零）。
- ARIMA建模的一般过程：
  - 平稳性检验
  - 白噪声检验
  - 定阶：
    - AR(p)偏自相关系数p阶截尾，自相关系数拖尾；
    - MA(p)自相关系数q阶截尾，偏自相关系数拖尾；
    - ARMA(p, q)自相关系数与偏自相关系数均拖尾；
    - 事实上，由于样本的随机性，很难确定是否是否真正截尾，因此引入两倍标准差范围；
    - 最初的d阶明显超过2倍标准差范围，而后几乎95%的（偏）自相关系数都落在2倍标准差范围以内，则d阶截尾。
  - 参数估计
  - 模型检验：
    - 模型的显著性检验：残差为白噪声，好的拟合模型能够提取观测序列中几乎所有样本的相关信息
    - 参数的显著性检验：每一个未知参数显著非零
  - 模型优化：AIC、SBC越小越好
 
 ## Jason博客
 
 ### [How to Create an ARIMA Model for Time Series Forecasting with Python](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)

 1. 求和自回归移动平均（ARIMA, AutoRegressive Integrated Moving Average）
 2. This acronym is descriptive, capturing the key aspects of the model itself. Briefly, they are:
   - AR: Autoregression. 使用当前观察点以及若干延迟时期观察点的依赖关系进行建模。
   - I: Integrated. 为使序列平稳而进行数据转化
   - MA: Moving Average. 使用当前观察点以及若干延迟时期观察点在移动平均模型上的残差的依赖关系进行建模。
 3. The parameters of the ARIMA model are defined as follows:
   - p: The number of lag observations included in the model, also called the lag order.
   - d: The number of times that the raw observations are differenced, also called the degree of differencing.
   - q: The size of the moving average window, also called the order of moving average.
   
- 数据是1901-1903年期间每月洗发水的销售量，每月销售量具有明显的上升趋势，因为这是非平稳序列，首先消除上升趋势。
- 模型定阶为ARIMA(5, 1, 0)
- 残差存在趋势，信息未完全提取
- 使用滚动预测，先划分训练集与测试集，使用训练集训练出第一个模型，在预测下一个值；之后将该真实值放入训练集重新开始训练。

### [How to Tune ARIMA Parameters in Python](https://machinelearningmastery.com/tune-arima-parameters-python/)

- 调节ARIMA模型参数
- walk-forward validation：下一个观察值会被加入训练集并更新模型
- 预设模型为ARIMA(4, 1, 0)
- disp: 详细信息展示
- transparams: 默认True；检查平稳性与可逆性
- trend: 偏置
- solver: 理解为解释器？['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell']

### [How to Grid Search ARIMA Model Hyperparameters with Python](https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/)

- 使用网格搜索调节超参，实际就是遍历不同的(p, d, q)组合
- 扩展：
  - 加入ACF、PACF
  - MSE可换成AIC或BIC
  - 可观察残差，是否符合高斯分布
  - 更新模型而不是重置模型
  - 建模前的数据检查，如平稳性等
 
### [Sensitivity Analysis of History Size to Forecast Skill with ARIMA in Python](https://machinelearningmastery.com/sensitivity-analysis-history-size-forecast-skill-arima-python/)

- How much history is required for a time series forecast model?
- 数据来源：墨尔本1981-1990年最低日气温，3650个观察点
- 消除季节性趋势：直接减去t-365点的值，存在的问题是没有考虑闰年，且第一年数据不可用
- 2-9年数据为训练集，第10年数据为测试集
- 分别使用1年、2年、...、8年数据作为训练集
- 结论：随着历史数据的增多，效果越好，但付出了运行时间的代价

### [How to Save an ARIMA Time Series Forecasting Model in Python](https://machinelearningmastery.com/save-arima-time-series-forecasting-model-python/)

- 保存ARIMA模型，由于存在Bug，本文主要解决bug
- 数据来源：1959年加州每日女生生育数

### [How to Make Out-of-Sample Forecasts with ARIMA in Python](https://machinelearningmastery.com/make-sample-forecasts-arima-python/)

- 数据来源：墨尔本1981-1990年最低日气温，3650个观察点
- 最后七个样本点作为数据以外的点
- forecast()预测下一时刻/时间段
- predict()可指定时间点/时间段预测
- 实现多步预测
