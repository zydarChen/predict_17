这个文件夹是曾经使用的代码。

1. RF_model.py : random_forest模型
2. fusai_dataprocessing.py ： 复赛的元数据处理，2017年3-6月数据，提取出时间特征。
3. stacking_rf_xgb.py ： 写的stacking代码，第二层用的是xgb，效果不好，但是二层的代码没毛病。
4. divid2_ronghe.py : 5个模型(static,best_mape,xgb,lightgbm,lstm),两两融合的结果，融合方式就是 求和除2。