#!/usr/bin/env python
# encoding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tslearn.clustering import TimeSeriesKMeans
from scipy.spatial.distance import pdist,squareform


def get_file_name(file_path, file_type):
    filename = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if os.path.splitext(file)[1] == file_type:
                filename.append(os.path.splitext(file)[0])
    return filename


def cluster_wf(cluster_number, inputseries, savePath):
    # euclidean
    km = TimeSeriesKMeans(n_clusters=cluster_number, metric="euclidean", max_iter=5,
                          random_state=0).fit(inputseries)
    y_pred = km.fit_predict(inputseries)
    print(y_pred)
    print(km.cluster_centers_.shape)
    # print(km.cluster_centers_)
    plt.figure()
    for yi in range(cluster_number):
        plt.subplot(2, 3, yi + 1)
        for xx in inputseries[y_pred == yi]:
            plt.plot(xx.ravel(), color=(59 / 255, 66 / 255, 62 / 255), alpha=.1)
        # (162 / 255, 135 / 255, 166 / 255)
        # (92 / 255, 35 / 255, 102 / 255)
        plt.plot(km.cluster_centers_[yi].ravel(), color=(131 / 255, 5 / 255, 24 / 255))
        plt.xlim(0, 100)
        plt.ylim(0, 5)
        plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                 transform=plt.gca().transAxes)
    plt.savefig(savePath + str(cluster_number) + "_clusters_wf.jpg")
    plt.show()
    return km, y_pred


def cluster_center_similarity(km, savePath):
    # Reshape cluster center to 2 dimensional
    cluster_center = np.reshape(km.cluster_centers_, (km.cluster_centers_.shape[0], km.cluster_centers_.shape[1]))
    print(cluster_center.shape)
    # Calculate dtw similarity martrix
    # ds = dtw.distance_matrix_fast(cluster_center)
    ds = squareform(pdist(cluster_center, metric='euclidean'))
    print(ds)
    # Plot
    sns.set_theme(style="white")
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(ds, dtype=bool))
    # Set up the matplotlib figure
    fig = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(ds, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, center=4.0)
    # ax.set_xticklabels(['A', 'B', 'C'], rotation=0)
    ax.set_xticklabels(["cluster " + str(i + 1) for i in range(cluster_number)], fontsize=15)
    ax.set_yticklabels(["cluster " + str(i + 1) for i in range(cluster_number)], fontsize=15)
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=15)
    plt.savefig(savePath + str(cluster_number) + "_clusters_similaity_matrix.jpg")
    plt.show()


if __name__ == '__main__':
    # 1. read all hist wfiuh file
    file_path = "D:/"
    savePath = "D:/results/"
    hist_nor_all = pd.read_csv(file_path + "hist_nor_all.csv")

    # 2. classify the WFIUH
    # parameter setting and result meaning please see tslearn tutorials
    # https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html
    cluster_number = 6
    # result 1: get cluster labels of each wfiuh
    km, cluster_label = cluster_wf(cluster_number, hist_nor_all, savePath)
    print(cluster_label)
    # result 2:different clustering results plot with different cluster_number
    cluster_center_similarity(km, savePath)

   







