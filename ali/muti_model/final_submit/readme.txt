final_submit文件夹放的是最终比赛使用的全部代码。

1. xgb.py : xgboost预测。
2. gbm.py : lightgbm预测。
3. lstm_lys_new.py : 使用lstm预测连续预测未来1个小时的结果。
4. 最优mape.py : 确定每条link，每小时的最优mape
5. static.py : 这是最优mape的一个特例，是我用 30个分钟时刻的mean、mode、median三个统计量加权平均的结果。
5. lstm_lys_Pred_onebyone.py: 使用lstm分别连续预测未来1个点、2个点……、30个点的结果；因为lstm单独预测8点02分这个点，比连续预测30个点中的8点02分这个点，预测的要好。所以，我们想替换“lstm_lys_new.py”预测结果中的前两个点。
6. instead_qian_2.py： 使用文件5中的结果，替换8点、15点、18点的前两个点的代码。
7. fusion_bylstm.py： 使用神经网络来做stacking模型融合，基础模型是 xgb、gbm、static、best_mape、lstm 5个模型，第二层模型用的是二层的权连接。