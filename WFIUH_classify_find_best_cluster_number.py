#!/usr/bin/env python
# encoding: utf-8

'''
---------------------------------------------------------
Project Name: 007-Hydrolib
File Name: analysis_8_clustering_wf_para.py
Author: HannahG
Create Date: 2021/12/13 14:12

@contact: guoyh.16b@igsnrr.ac.cn
@desc:
---------------------------------------------------------
'''


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import genfromtxt
from gap_statistic import optimalK


def plot_results(savePath, gap_df):
    """
    Plots the results of the last run optimal K search procedure.
    Four plots are printed:
    (1) A plot of the Gap value - as defined in the original Tibshirani et
    al paper - versus n, the number of clusters.
    (2) A plot of diff versus n, the number of clusters, where diff =
    Gap(k) - Gap(k+1) + s_{k+1}. The original Tibshirani et al paper
    recommends choosing the smallest k such that this measure is positive.
    (3) A plot of the Gap* value - a variant of the Gap statistic suggested
    in "A comparison of Gap statistic definitions with and with-out
    logarithm function" [https://core.ac.uk/download/pdf/12172514.pdf],
    which simply removes the logarithm operation from the Gap calculation -
    versus n, the number of clusters.
    (4) A plot of the diff* value versus n, the number of clusters. diff*
    corresponds to the aforementioned diff value for the case of Gap*.
    """


    # Gap values plot
    plt.plot(gap_df.n_clusters, gap_df.gap_value, linewidth=3)
    plt.scatter(
        gap_df[gap_df.n_clusters == n_clusters].n_clusters,
        gap_df[gap_df.n_clusters == n_clusters].gap_value,
        s=250,
        c="r",
    )
    plt.grid(True)
    plt.xlabel("Cluster Count")
    plt.ylabel("Gap Value")
    plt.title("Gap Values by Cluster Count")
    plt.savefig(savePath + "kmeans_gapstat_gap.svg")
    plt.show()

    # diff plot
    plt.plot(gap_df.n_clusters, gap_df["diff"], linewidth=3)
    plt.grid(True)
    plt.xlabel("Cluster Count")
    plt.ylabel("Diff Value")
    plt.title("Diff Values by Cluster Count")
    plt.savefig(savePath + "kmeans_gapstat_diff.svg")
    plt.show()

    # Gap* plot
    max_ix = gap_df[gap_df["gap*"] == gap_df["gap*"].max()].index[0]
    plt.plot(gap_df.n_clusters, gap_df["gap*"], linewidth=3)
    plt.scatter(
        gap_df.loc[max_ix]["n_clusters"],
        gap_df.loc[max_ix]["gap*"],
        s=250,
        c="r",
    )
    plt.grid(True)
    plt.xlabel("Cluster Count")
    plt.ylabel("Gap* Value")
    plt.title("Gap* Values by Cluster Count")
    plt.savefig(savePath + "kmeans_gapstat_gap_.svg")
    plt.show()

    # diff* plot
    plt.plot(gap_df.n_clusters, gap_df["diff*"], linewidth=3)
    plt.grid(True)
    plt.xlabel("Cluster Count")
    plt.ylabel("Diff* Value")
    plt.title("Diff* Values by Cluster Count")
    plt.savefig(savePath + "kmeans_gapstat_diff_.svg")
    plt.show()


if __name__ == '__main__':
    file_path = "D:/"
    savePath = "D:/results/"
    hist_nor_all = genfromtxt(file_path + 'hist_nor_all.csv', delimiter=',')
    print(hist_nor_all)

    # Find best cluster number
    # see https://github.com/milesgranger/gap_statistic
    optimalK = optimalK.OptimalK(n_jobs=4, parallel_backend='joblib')
    n_clusters = optimalK(hist_nor_all, cluster_array=np.arange(3, 11))
    optimalK.gap_df.head()
    sns.set_theme(style="white")
    print(n_clusters)
    print(optimalK.gap_df)
    gap_df = optimalK.gap_df
    plot_results(savePath, gap_df)








