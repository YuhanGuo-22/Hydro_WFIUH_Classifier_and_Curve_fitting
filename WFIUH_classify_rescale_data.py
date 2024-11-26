#!/usr/bin/env python
# encoding: utf-8


import numpy as np
import pandas as pd
import os
import itertools
import math


def get_file_name(file_path, file_type):
    filename = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if os.path.splitext(file)[1] == file_type:
                filename.append(os.path.splitext(file)[0])
    return filename


def wf2pdf(WF_x, WF_y, interval=30, Tinv=0.03):
    # v = 30  # 单位为km/hour
    Total_wf_x = []
    for i in range(len(WF_x)):
        for n in itertools.repeat(WF_x[i], WF_y[i]):
            Total_wf_x.append(n)
    # print(Total_wf_x)
    # bins = np.arange(math.floor(min(Total_wf_x)), math.ceil(max(Total_wf_x)), Tinv)
    bins = np.linspace(math.floor(min(Total_wf_x)), math.ceil(max(Total_wf_x)), num=interval, endpoint=True, retstep=False, dtype=None)
    # print(bins)
    hist, bin_edge = np.histogram(Total_wf_x, bins=bins)
    hist = np.insert(hist, 0, 0)
    # print(hist)
    hist_nor = []
    hist_total = np.sum(hist, axis=0)
    for i in range(len(hist)):
        hist_nor.append(hist[i]/hist.sum() * interval)
    # print(bin_edge)
    # print(hist_nor)
    # WF = pd.DataFrame({"bin": list(bin_edge), "hist": hist_nor})
    # ax = sns.barplot(x="bin", y="hist", data=WF)
    # ticks_spacing = 10
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks_spacing))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.set_title("histogram")
    # ax.set_xlabel("Scaled distance from outlet")
    # ax.set_ylabel("Densities")
    # plt.show()
    return hist_nor, bin_edge


if __name__ == '__main__':
    # 1. read all the wfiuh from a file
    file_path = "I:/wfiuh/"
    savePath = 'I:/wfiuh/'
    file_type = '.csv'
    filename = get_file_name(file_path, file_type)
    print(filename)

    # 2. rescaled all the wfiuh into same length and interval
    # note that the sequence of hist_nor_all wfiuh is corresponding to the filename list
    catName = []
    hist_nor_all = []
    for n in range(0, len(filename)):
        wfiuh = pd.read_csv(file_path + filename[n] + ".csv", engine='python', skiprows=0)
        wfiuh["flowTime_standard"] = wfiuh["flowTime"].apply(lambda x:  x / wfiuh["flowTime"].max())
        x = wfiuh["flowTime_standard"]
        y = wfiuh["cells"]
        # y = wfiuh["frequency"]
        hist_nor, bin_edge = wf2pdf(x, y)
        hist_nor_all.append(np.array(hist_nor).reshape((len(hist_nor), 1)))
        catName.append(filename[n].split("-")[0])
    print(catName)
    hist_nor_all = np.array(hist_nor_all)
    print(hist_nor_all.shape)

    hist_nor_all_save = np.reshape(hist_nor_all, (hist_nor_all.shape[0], hist_nor_all.shape[1]))
    print(hist_nor_all_save)
    np.savetxt(savePath + "hist_nor_all.csv", hist_nor_all_save, delimiter=",")






