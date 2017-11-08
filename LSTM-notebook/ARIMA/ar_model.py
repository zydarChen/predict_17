# coding=utf-8
#arma模型中的y值要为float类型，而不是int
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import arma_order_select_ic
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.sandbox.regression.kernridgeregress_class import plt_closeall


class ARModel():

    def testStationary(self, data, max_lag=None):
        # 原假设为信号为非平稳
        '''
        adf is -4.85811662327 ,pvalue is 4.204754858e-05.usedlag=44
        nobs is 17627,icbest is -320762.899743
        '''
        if max_lag == None:
            adf_result = adfuller(data)
        else:
            adf_result = adfuller(data, maxlag=max_lag)

        if adf_result[1] < 0.05:
            return True
        elif adf_result[1] < 0.1:
            print 'adf is {0} ,pvalue is {1}.usedlag={2}'.format(adf_result[0], adf_result[1], adf_result[2])
            print 'nobs is {0},icbest is {1}'.format(adf_result[3], adf_result[5])
            return True
        else:
            print 'adf is {0} ,pvalue is {1}.usedlag={2}'.format(adf_result[0], adf_result[1], adf_result[2])
            print 'nobs is {0},icbest is {1}'.format(adf_result[3], adf_result[5])
            print 'data is not stationary'
            return False

    def testRandomness(self, data, max_lag=None):
        # 原假设为信号为白噪声
        corr_result = acorr_ljungbox(data, lags=max_lag)
        if corr_result[1][-1] < 0.05:
            # print 'test randomness lbvalue is {0}, pvalue is
            # {1}'.format(corr_result[0],corr_result[1])
            print 'data is not white noise'
            return False
        else:
            # print 'test randomness lbvalue is {0}, pvalue is
            # {1}'.format(corr_result[0],corr_result[1])
            return True

    def constructFeature(self, df, x_col_name,y_col_name, window_len, moving_len, p, q):
        print 'start construct feature'
        length = len(df)
        rows = (length - window_len) / moving_len
        feature_array = []
        for index in xrange(rows):
            x_temp = df.index[index * moving_len:index *
                             moving_len + window_len].values
            y_temp = df.iloc[index * moving_len:index *
                             moving_len + window_len][y_col_name].values
            if self.testStationary(y_temp) and (not self.testRandomness(y_temp)):
                #           print df_temp.head()
                
                array = self.trainARMAModel(y_temp, x_temp, p, q)
                if array != None:
                    array.append(x_temp[0])  # 添加时间索引
                    feature_array.append(array)
            else:
                print 'y_temp is not stationary or is random'
        print 'after a series data shape is{0}'.format(np.array(feature_array).shape)
        return feature_array

    def constructSARMAXFeature(self, df, x_col_name,y_col_name, window_len, moving_len, order, seasonal_order):
        print 'start construct feature'
        length = len(df)
        rows = (length - window_len) / moving_len - 1
        feature_array = []
        for index in range(rows):
            y_data = df.iloc[index * moving_len:index *
                             moving_len + window_len][y_col_name].values
            x_data = df.iloc[index * moving_len:index *
                             moving_len + window_len][x_col_name].values
            if self.testStationary(y_data) and (not self.testRandomness(y_data)):
                #           print df_temp.head()
                try:
                    array = self.trainSARIMAXModel(
                        y_data, order, seasonal_order)
                    if array != None:
                        array.append(x_data[0])  # 添加时间索引
                        feature_array.append(array)
                except Exception, e:
                    print e
            else:
                print 'x_temp is not stationary or is random'
        print 'after a series data shape is{0}'.format(np.array(feature_array).shape)
        return feature_array

   # df is an dateframe with datetime index
    def trainARMAModel(self, data, dates, p, q):
        try:
            start = time.time()
            arma = ARMA(data, order=(p, q), dates=dates).fit(disp=False)
            end = time.time()
            print 'train ARMA model cost %fs' % ((end - start))
            return arma.params.values[:-q].tolist()  # 只保留ar系数和常数
        except Exception, e:
            print e
            print 'Exception occur during training arma model at {0}'.format(dates[0])

    def trainSARIMAXModel(self, y_data, order, seasonal_order):
        """ order=(p,d,q) ,seasonal_order=(P,D,Q,S)"""
        try:
            start = time.time()
            arma = SARIMAX(y_data, order=order, seasonal_order=seasonal_order)
            arma_fit = arma.fit(disp=False)
            end = time.time()
            print 'train saramx model cost %fs' % (end - start)
            # 只返回ar(p)系数
            return arma_fit.params[:order[0]].tolist()
        except Exception, e:
            print e
            print 'train arimax model process'

    def plotACF(self, df, y_col_name, year, start_month, window_len, moving_len, acf_lag, pacf_lag, datetime_delimiter):
        df_temp = df[df['date'] > year + datetime_delimiter + start_month]
        print 'the data length for test stationary is %d' % (len(df_temp))
        length = len(df_temp)
        rows = (length - window_len) / moving_len
        # ix按照行索引寻找数据，按照行索引时先要把重新排序索引
        for index in range(0, 2):
            df_temp = df.iloc[
                index * moving_len:index * moving_len + window_len]
            print 'df temp length is %d' % (len(df_temp))
            if self.testStationary(df_temp[y_col_name].values) and not (self.testRandomness(df_temp[y_col_name].values)):
                fig = plot_acf(df_temp[y_col_name], lags=acf_lag)
                fig2 = plot_pacf(df_temp[y_col_name], lags=pacf_lag)
        plt.show()

    def plotRawDataACF(self, y_data, acf_lag, pacf_lag, window_len, moving_len, plot_flag=True):
        length = len(y_data)
        rows = (length - window_len) / moving_len
        print 'rows=%d' % (rows)
        for index in range(rows):
            y_temp = y_data[index * moving_len:index * moving_len + window_len]
            if plot_flag:
                if self.testStationary(y_temp) and not (self.testRandomness(y_temp)):
                    fig = plot_acf(y_temp, lags=acf_lag)
                    fig2 = plot_pacf(y_temp, lags=pacf_lag)
                    plt.show()
            else:
                res = arma_order_select_ic(
                    y_temp, 4, 1, fit_kw={'method': 'css'})
                print res.bic_min_order
        
    def VerifyOrder(self, df,y_col_name,window_len,moving_len, p, q, getOrder=True):
        """ 若getOrder=True,则返回最优的Arma阶数，否则则对特定的p,q根据aic,bic选择最优"""
        length = len(df)
        rows = (length - window_len) / moving_len
        print 'rows is ', rows
        for index in range(rows):
            df_temp = df.iloc[index * moving_len:index * moving_len + window_len]
            if getOrder:                
                try:
                    res = arma_order_select_ic(df_temp[y_col_name], p, q, fit_kw={'method': 'css'})
                    print 'min order', res.bic_min_order
                except Exception, e:
                    print e
            else:
                try:
                    arma = ARMA(df_temp[y_col_name], order=(
                        p, q))                                
                    arma_fit = arma.fit(disp=False)
                    print 'arma model aci is %f,bic is %f' % (arma_fit.aic, arma_fit.bic)

                    # 检验模型参数的显著性
                    # print 'model coeffs params is',arma_fit.pvalues
                    pvalues = arma_fit.pvalues
                    if len(pvalues[pvalues > 0.05]) > 1:
                        print 'arma model coeffs is not outstanding,there is %d coeffs greater then 0.05' % (len(pvalues[pvalues > 0.05]) - 1)
                        # print pvalues
                    # 检验模型的显著性，及模型的残差序列为白噪声序列，信息已经充分提取
                    pred_valus = arma.predict(
                        params=arma_fit.params, dynamic=False)
                    print 'pred length is %d' % (len(pred_valus))
                    print 'the exact length is %d' % (df_temp[y_col_name].values[window_len - len(pred_valus):])
                    residuals = df_temp[y_col_name].values[
                        window_len - len(pred_valus):] - pred_valus
                    if self.testRandomness(residuals):
                        print 'the residuals is white noise' + '!' * 20
                    else:
                        print 'the residuals is not white noise'
                except Exception, e:
                    print e

        # print 'the aic,bic for order({0},{1}) is
        # {2},{3}'.format(p,q,np.mean(aic_list),np.mean(bic_list))
    def VerifySARMAXOrder(self, df, y_col_name, window_len, moving_len, order, sensonal_order):
        """ df contains all the data,"""
        length = len(df)
        rows = (length - window_len) / moving_len - 1
        for index in range(rows):
            df_temp = df.iloc[
                index * moving_len:index * moving_len + window_len]
        # print df_temp.head()
        # print 'df temp length is %d' %(len(df_temp))
            try:
                arma = SARIMAX(df_temp[y_col_name], order=order,
                               sensonal_order=sensonal_order)
                arma_fit = arma.fit(disp=False)
                print 'sarimax model aci is %f,bic is %f' % (arma_fit.aic, arma_fit.bic)

                # 检验模型参数的显著性
                # print 'model coeffs params is',arma_fit.pvalues
                pvalues = arma_fit.pvalues
                if len(pvalues[pvalues > 0.05]) > 1:
                    print 'arma model coeffs is not outstanding,there is %d coeffs greater then 0.05' % (len(pvalues[pvalues > 0.05]) - 1)
                    # print pvalues
                # 检验模型的显著性，及模型的残差序列为白噪声序列，信息已经充分提取
                # pred_valus = arma.predict(params=arma_fit.params, dynamic=False)
                pred_valus = arma_fit.predict()
                # plt.plot(df_temp.index,df_temp[y_col_name].values,'bs',df_temp.index,pred_valus,'ro')
                print 'pred length is %d' % (len(pred_valus))
                print 'the exact length is %d' % (len(df_temp[y_col_name].values[window_len - len(pred_valus):]))
                residuals = df_temp[y_col_name].values[
                    window_len - len(pred_valus):] - pred_valus
                if self.testRandomness(residuals):
                    print 'the residuals is white noise' + '!' * 20
                else:
                    print 'the residuals is not white noise'

            except Exception, e:
                print e
            # plt.show()

        # print 'the aic,bic for order({0},{1}) is
        # {2},{3}'.format(p,q,np.mean(aic_list),np.mean(bic_list))
        

