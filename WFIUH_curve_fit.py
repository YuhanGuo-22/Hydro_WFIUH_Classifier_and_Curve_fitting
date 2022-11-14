#!/usr/bin/env python
# encoding: utf-8


'''
---------------------------------------------------------
@desc:This .py file is used to fit the wfiuh with different distribution
      Here we take Beta distribution (both cdf and pdf) as an example
      note that the pdf and cdf both have the same parameter, and the pdf is discontinuous,
      so the best solution is that you first fit the cdf curve and then use the fitted parameter to evaluate pdf
---------------------------------------------------------
'''


import os
import pandas as pd
import numpy as np
import hydroeval as he
from scipy.optimize import curve_fit
import scipy.special as sc
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import traceback


def Beta_distribution_CDF(x,a,b):
    return sc.betainc(a, b, x)


def Beta_distribution_PDF(x, a, b):
    # return sc.betainc(a, b, x)
    return x ** (a - 1) * (1 - x) ** (b - 1) / sc.beta(a, b)


def get_file_name(file_path, file_type):
    filename = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if os.path.splitext(file)[1] == file_type:
                filename.append(os.path.splitext(file)[0])
    return filename


if __name__ == '__main__':
    # 1. find all the wfiuh csv file in a filepath
    file_path = "D:/wfiuh/"
    savePath = 'D:/wfiuh/'
    file_type = '.csv'
    filename = get_file_name(file_path, file_type)
    print(filename)

    for i in range(0, len(filename)):
        try:
            print(i)
            # print(i)
            # 1.1 Read wfiuh
            wfiuh = pd.read_csv(file_path + "/" + filename[i] + ".csv", engine='python', skiprows=0)
            wfiuh["frequency_acc"] = wfiuh["frequency"].cumsum()
            x = wfiuh["flowTime"]
            y = wfiuh["frequency_acc"]
          
            # 1.2 Curve fitting
            # sigma can be used in fitting CDF curve
            # Using parameter sigma to force fitted curve going through (0,0) and (1,1)
            sigma = np.ones(len(x))
            sigma[[0, -1]] = 0.01
            popt, pcov = curve_fit(Beta_distribution_CDF, np.array(x), np.array(y), sigma=sigma, maxfev=2000)

            # 1.3 Calculate evaluate index
            r2 = r2_score(np.array(y), Beta_distribution_CDF(np.array(x), *popt))
            # squared: bool, default = True
            #    If True returns MSE value,if False returns RMSE value.
            RMSE = mean_squared_error(np.array(y), Beta_distribution_CDF(np.array(x), *popt), squared=False)
            kge, r, alpha, beta = he.evaluator(he.kge, Beta_distribution_CDF(np.array(x), *popt), np.array(y))
        
        except Exception as e:
            traceback.print_exc()
            continue
       

