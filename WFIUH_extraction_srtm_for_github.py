#!/usr/bin/env python
# encoding: utf-8
"""
@author: Hannah Guo
@File : 000-WF_12个流域_srtm_TP.py
@Time : 2022/6/7 10:59
@Site : 
@Desc:
"""

import os
import traceback
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pysheds.grid import Grid
import pandas as pd
import fiona
from scipy.sparse import csgraph
import scipy
import warnings
import numpy as np
import rasterio
import math
import seaborn as sns
import numba as nb
import glob
import os
from queue import Queue
from threading import Thread
from time import time
from scipy import stats
import networkx as nx
from pysheds.view import Raster
from pysheds.view import BaseViewFinder, RegularViewFinder, IrregularViewFinder


def plot_flowdir(save_path, cat_Name, catches, dirmap):
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_alpha(0)

    plt.imshow(catches, cmap='viridis', zorder=2)
    boundaries = ([0] + sorted(list(dirmap)))
    plt.colorbar(boundaries=boundaries,
                 values=sorted(dirmap))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flow direction grid')
    plt.grid(zorder=-1)
    plt.tight_layout()
    plt.savefig(save_path + cat_Name + '_flow_direction.png', bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_river(acc_Thres, savePath, cat_Name, xy, new_xy, branches):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    plt.grid('on', zorder=1)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title("River network (>" + str(np.round(acc_Thres, 2)) + "accumulation)")
    # plt.xlim(grid.bbox[0], grid.bbox[2])
    # plt.ylim(grid.bbox[1], grid.bbox[3])
    ax.set_aspect('equal')

    plt.scatter(xy[0][0], xy[0][1], marker='x', c='b', s=100, label='original')
    plt.scatter(new_xy[0][0], new_xy[0][1], marker='x', c='r', s=100, label='snapped')

    for branch in branches['features']:
        line = np.asarray(branch['geometry']['coordinates'])
        plt.plot(line[:, 0], line[:, 1], zorder=2)

    plt.savefig(savePath + cat_Name + '_flow_river.png')
    # plt.show()
    plt.close()


def plot_flowacc(savePath, cat_Name, xy, new_xy, grid, branches):
    fig, ax = plt.subplots(figsize=(8, 6))
    # fig.patch.set_alpha(0)
    plt.grid('on', zorder=1)
    acc_img = np.where(grid.mask, grid.acc + 1, np.nan)
    im = plt.imshow(acc_img, extent=grid.extent, zorder=2,
                    cmap='cubehelix',
                    norm=colors.LogNorm(1, grid.acc.max()))
    plt.colorbar(im, ax=ax, label='Upstream Cells')

    for branch in branches['features']:
        line = np.asarray(branch['geometry']['coordinates'])
        plt.plot(line[:, 0], line[:, 1], zorder=3)

    plt.scatter(xy[0][0], xy[0][1], marker='x', c='b', s=100, label='original', zorder=4)
    plt.scatter(new_xy[0][0], new_xy[0][1], marker='x', c='r', s=100, label='snapped', zorder=5)

    plt.title('Flow Accumulation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(savePath + cat_Name + '_flow_accumulation.png', bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_flowdis(savePath, cat_Name, grid):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0)
    plt.grid('on', zorder=1)
    grid.clip_to('catch')
    flow_dist = grid.view('dist', nodata=np.nan)
    im = ax.imshow(flow_dist, extent=grid.extent, zorder=2,
                   cmap='cubehelix_r')
    plt.colorbar(im, ax=ax, label='Distance to outlet (cells)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flow Distance')
    plt.savefig(savePath + cat_Name + '_flow_distance.png', bbox_inches='tight')
    # plt.show()
    plt.close()


def _flatten_fdir(fdir, flat_idx, dirmap, copy=False):
    # WARNING: This modifies fdir in place if copy is set to False!
    if copy:
        fdir = fdir.copy()
    shape = fdir.shape
    go_to = (
        0 - shape[1],
        1 - shape[1],
        1 + 0,
        1 + shape[1],
        0 + shape[1],
        -1 + shape[1],
        -1 + 0,
        -1 - shape[1]
    )
    gotomap = dict(zip(dirmap, go_to))
    for k, v in gotomap.items():
        fdir[fdir == k] = v
    fdir.flat[flat_idx] += flat_idx


def _unflatten_fdir(fdir, flat_idx, dirmap):
    shape = fdir.shape
    go_to = (
        0 - shape[1],
        1 - shape[1],
        1 + 0,
        1 + shape[1],
        0 + shape[1],
        -1 + shape[1],
        -1 + 0,
        -1 - shape[1]
    )
    gotomap = dict(zip(go_to, dirmap))
    fdir.flat[flat_idx] -= flat_idx
    for k, v in gotomap.items():
        fdir[fdir == k] = v
   
    
def _construct_matching(fdir, flat_idx, dirmap, fdir_flattened=False):
    # TODO: Maybe fdir should be flattened outside this function
    if not fdir_flattened:
        _flatten_fdir(fdir, flat_idx, dirmap)
    startnodes = flat_idx
    endnodes = fdir.flat[flat_idx]
    return startnodes, endnodes


def cal_flow_distance(x, y, fdir, weights=None, dirmap=None, nodata_in=None,
                  nodata_out=0, out_name='dist', routing='d8', method='shortest',
                  inplace=True, xytype='index', apply_mask=True, ignore_metadata=False):
    xmin, ymin, xmax, ymax = fdir.bbox
    if xytype in ('label', 'coordinate'):
        if (x < xmin) or (x > xmax) or (y < ymin) or (y > ymax):
            raise ValueError('Pour point ({}, {}) is out of bounds for dataset with bbox {}.'
                             .format(x, y, (xmin, ymin, xmax, ymax)))
    elif xytype == 'index':
        if (x < 0) or (y < 0) or (x >= fdir.shape[1]) or (y >= fdir.shape[0]):
            raise ValueError('Pour point ({}, {}) is out of bounds for dataset with shape {}.'
                             .format(x, y, fdir.shape))
    # Construct flat index onto flow direction array
    domain = np.arange(fdir.size)
    fdir_orig_type = fdir.dtype
    if nodata_in is None:
        nodata_cells = np.zeros_like(fdir).astype(bool)
    else:
        if np.isnan(nodata_in):
            nodata_cells = (np.isnan(fdir))
        else:
            nodata_cells = (fdir == nodata_in)
    try:
        mintype = np.min_scalar_type(fdir.size)
        fdir = fdir.astype(mintype)
        domain = domain.astype(mintype)
        startnodes, endnodes = _construct_matching(fdir, domain,
                                                        dirmap=dirmap)
        # print(startnodes)
        # print(fdir.shape)
        if weights is not None:
            weights = weights.ravel()
            assert (weights.size == startnodes.size)
            assert (weights.size == endnodes.size)
        else:
            assert (startnodes.size == endnodes.size)
            weights = (~nodata_cells).ravel().astype('float32')
        C = scipy.sparse.lil_matrix((fdir.size, fdir.size))
        for i, j, w in zip(startnodes, endnodes, weights):
            C[i, j] = w
        C = C.tocsr()
        # print(C)
        xyindex = np.ravel_multi_index((y, x), fdir.shape)
        # print(xyindex)
        dist, predecessors = csgraph.shortest_path(C, indices=[xyindex], directed=False, return_predecessors=True)
        dist[~np.isfinite(dist)] = nodata_out
        dist = dist.ravel()
        dist = dist.reshape(fdir.shape)
    except:
        raise
    finally:
        _unflatten_fdir(fdir, domain, dirmap)
        fdir = fdir.astype(fdir_orig_type)
    # Prepare output
    return dist, predecessors, xyindex

@nb.jit(fastmath=True)
def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path


@nb.jit(fastmath=True)
def get_velocity_sum(catShape, shortest_paths,velocity):
    velocity_sum_all = np.zeros(len(shortest_paths), dtype=float)
    for n in range(len(shortest_paths)):
        multi_index = np.unravel_index(shortest_paths[n], catShape)
        velocity_temp = velocity[multi_index[0], multi_index[1]]
        if velocity_temp == -999:
            velocity_temp = 0
        velocity_sum_all[n] = velocity_temp
    velocity_sum = sum(velocity_sum_all)
    return velocity_sum


@nb.jit(fastmath=True)
def calculate_avg_velocity(catShape, predecessors, velocity):
    velocity_average = np.zeros_like(velocity)
    for k in range(0, predecessors.shape[1]):
        shortest_paths = get_path(predecessors, 0, k)
        velocity_sum = get_velocity_sum(catShape, shortest_paths,velocity)
        start_index = np.unravel_index(k, catShape)
        velocity_average[start_index[0], start_index[1]] = velocity_sum / len(shortest_paths)
    return velocity_average


def calculate_WFIUH(cat_Name, filePath, savePath, x, y):
    try:
        start_time = time()
        # 1. Load flow direction tiff
        grid = Grid.from_raster(filePath + "1-sub-dir-clip/" + cat_Name + '.tif', data_name='catch')
        # N    NE    E    SE    S    SW    W    NW
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        catches = grid.view('catch', nodata=np.nan)
        # plot_flowdir(savePath, cat_Name, catches, dirmap)
        
        # 2. Compute flow accumulation
        grid.accumulation(data='catch', dirmap=dirmap, out_name='acc', nodata_in=0)
        
        # 3. Snap the station location to the pour point
        # Find an approximate threshold accumulation value that captures that structure
        # (using percentile way mentioned by MDB)
        acc_data = pd.DataFrame(grid.acc.ravel())
        acc_data = acc_data[acc_data[0] != 0]
        acc_Thres = np.percentile(acc_data, 96)
        # Snap the station location to the pour point
        # Specify pour point (greatest accumulation)
        xy = np.column_stack([x, y])
        new_xy = grid.snap_to_mask(grid.acc > acc_Thres, xy, return_dist=False)
        # print(new_xy)
        
        # 4. River network extraction
        # Get the river network of the catchments
        branches = grid.extract_river_network('catch', 'acc', threshold=acc_Thres, dirmap=dirmap)
        plot_river(acc_Thres, savePath, cat_Name, xy, new_xy, branches)
        
        schema = {
            'geometry': 'LineString',
            'properties': {}
        }
        
        with fiona.open(savePath + "River/" + cat_Name + '.shp', 'w',
                        driver='ESRI Shapefile',
                        crs=grid.crs.srs,
                        schema=schema) as c:
            i = 0
            for branch in branches['features']:
                rec = {}
                rec['geometry'] = branch['geometry']
                rec['properties'] = {}
                rec['id'] = str(i)
                c.write(rec)
                i += 1
        # grid.to_raster(branches, save_path + "River/" + cat_Name + ".tif", blockxsize=16, blockysize=16)
        
        # 5. Extract flow distance
        # Compute flow distance
        grid.flow_distance(data='catch', x=new_xy[0][0], y=new_xy[0][1], dirmap=dirmap, out_name="dist", xytype='label')
        # Get flow distance array
        dists = grid.view("dist")
        grid.to_raster(dists, savePath + "Flowdistance/" + cat_Name + ".tif", blockxsize=16, blockysize=16)
        
        # 6. Extract sum of velocity along the flow path
        # namely calculate the weighted flow distacne, and the weights is the gridded velocity map
        # Compute weighted distance to outlet
        with rasterio.open(filePath + "7-sub-Manning-v-transfer/" + str(cat_Name) + '.tif') as src:
            velocity = src.read(1)
            # exclude velocity equals 0,  consider slope is 1 degree and the roughness is maximum
            # ((0.01 ** (2 / 3) * 1.7455 ** (1 / 2))) / (0.6) = 0.1022
            velocity[velocity == 0] = 0.1022
            # velocity_masked = np.ma.masked_array(velocity, mask=(velocity == -999))
            # print(velocity_masked.shape)
        xy_index = grid.nearest_cell(new_xy[0][0], new_xy[0][1])
        dist, predecessors, xy_node = cal_flow_distance(fdir=grid.catch, x=xy_index[0], y=xy_index[1], dirmap=dirmap)
        # Find the shortest path from each point to the exit point. k refers to the kth point in the watershed.
        # Since the shortest path solution sets the outlet, the predecessors become an array of 1 row and k columns
        catShape = grid.catch.shape
        velocity_average = calculate_avg_velocity(catShape, predecessors, velocity)
        
        # 7. Calculate the flow time distribution according to the flow distance and flow velocity
        # velocity mean along flow path(m/s) = velocity mean along flow sum(m/s) / flow distance(cells)
        # flow time = flow distance(m) / velocity mean along flow path(m/s)
        distance_m = math.sqrt(resolution ** 2 + resolution ** 2) * dists
        
        # 8. Calculate the WFIUH
        # time interval index，3600 represents 60s*20, 20 mins
        # velocity_masked = np.ma.masked_array(velocity_average, mask=(velocity_average == 0))
        # distance_m_masked = distance_m_masked
        time_index = 1200
        flowTime = (distance_m / (velocity_average)) / time_index
        # np.savetxt(filePath + "7-sub-WFIUH-new1/test4.csv", flowTime, delimiter=",")
        flowTime = np.where(flowTime == 0, -999, flowTime)
        flowTime = np.where(np.isnan(flowTime), -999, flowTime)
        # np.savetxt(filePath + "7-sub-WFIUH-new/test.csv", flowTime, delimiter=",")
        # 2. Since some mesh confluence time will be very large,
        # it is necessary to limit the maximum confluence time
        flowTime_max = (np.max(distance_m) / (np.mean(velocity_average[velocity_average != 0]))) / time_index + 40
        # flowTime_max = np.max(flowTime)
        # # 3. define interval
        x = np.linspace(0, int(flowTime_max), int(flowTime_max) + 1)
        W = np.histogram(flowTime[flowTime != -999], bins=len(x), range=(0, flowTime_max))
        x_hour = W[1] / (3600.0 / time_index)
        cells = np.insert(W[0], 0, 0, axis=0)
        # 4. save
        WFIUH = pd.DataFrame({"flowTime": x_hour, "cells": cells})
        WFIUH.to_csv(savePath + cat_Name + "_wfiuh_" + str(resolution) + ".csv")
        sns.lineplot(x=x_hour, y=cells)
        plt.xlabel("Time (hour)")
        plt.ylabel("Number of cells at distance x from outlet")
        plt.title("WFIUH")
        plt.savefig(savePath + cat_Name + '_wfiuh.jpg')
        # plt.show()
        plt.close()

        # If the non-zero grid number of the confluence distance is within the 5% range of
        # the non-zero grid number of the flow direction, it is regarded as the effective confluence distance
        catch_nonzero = np.count_nonzero(grid.catch, axis=None)
        dist_nonzero = np.count_nonzero(grid.dist, axis=None)
        velocity_average_nonzero = np.count_nonzero(velocity_average, axis=None)
        print("-----------------------------------")
        # print(catch_nonzero)
        # print(dist_nonzero)
        # print(velocity_average_nonzero)
        # print(grid.catch.shape)
        # print(grid.dist.shape)
        print(cat_Name + " finished and runtime: --- %s seconds ---" % (time() - start_time))
        
        if dist_nonzero < catch_nonzero * 0.85 or dist_nonzero > catch_nonzero * 1.15 or \
                velocity_average_nonzero < catch_nonzero * 0.85 or velocity_average_nonzero > catch_nonzero * 1.15:
            if os.path.exists(savePath + cat_Name + '_flow_direction.png'):
                os.remove(savePath + cat_Name + '_flow_direction.png')
            if os.path.exists(savePath + cat_Name + '_wfiuh.jpg'):
                os.remove(savePath + cat_Name + '_wfiuh.jpg')
            if os.path.exists(savePath + cat_Name + '_flow_river.png'):
                os.remove(savePath + cat_Name + '_flow_river.png')
            if os.path.exists(savePath + cat_Name + "_wfiuh_" + str(resolution) + ".csv"):
                os.remove(savePath + cat_Name + "_wfiuh_" + str(resolution) + ".csv")
            
            if os.path.exists(savePath + "Flowdistance/" + cat_Name + ".tif"):
                os.remove(savePath + "Flowdistance/" + cat_Name + ".tif")
            if os.path.exists(savePath + "River/" + cat_Name + '.shp'):
                os.remove(savePath + "River/" + cat_Name + '.shp')
            if os.path.exists(savePath + "River/" + cat_Name + '.cpg'):
                os.remove(savePath + "River/" + cat_Name + '.cpg')
            if os.path.exists(savePath + "River/" + cat_Name + '.dbf'):
                os.remove(savePath + "River/" + cat_Name + '.dbf')
            if os.path.exists(savePath + "River/" + cat_Name + '.prj'):
                os.remove(savePath + "River/" + cat_Name + '.prj')
            if os.path.exists(savePath + "River/" + cat_Name + '.shp'):
                os.remove(savePath + "River/" + cat_Name + '.shp')
            if os.path.exists(savePath + "River/" + cat_Name + '.shx'):
                os.remove(savePath + "River/" + cat_Name + '.shx')
            
            print(cat_Name + "  it has been deleted!")
    
    except Exception as e:
        traceback.print_exc()
        if os.path.exists(savePath + cat_Name + '_flow_direction.png'):
            os.remove(savePath + cat_Name + '_flow_direction.png')
        if os.path.exists(savePath + cat_Name + '_wfiuh.jpg'):
            os.remove(savePath + cat_Name + '_wfiuh.jpg')
        if os.path.exists(savePath + cat_Name + "_wfiuh_" + str(resolution) + ".csv"):
            os.remove(savePath + cat_Name + "_wfiuh_" + str(resolution) + ".csv")

        if os.path.exists(savePath + "Flowdistance/" + cat_Name + ".tif"):
            os.remove(savePath + "Flowdistance/" + cat_Name + ".tif")
        if os.path.exists(savePath + "River/" + cat_Name + '.shp'):
            os.remove(savePath + "River/" + cat_Name + '.shp')
        if os.path.exists(savePath + "River/" + cat_Name + '.cpg'):
            os.remove(savePath + "River/" + cat_Name + '.cpg')
        if os.path.exists(savePath + "River/" + cat_Name + '.dbf'):
            os.remove(savePath + "River/" + cat_Name + '.dbf')
        if os.path.exists(savePath + "River/" + cat_Name + '.prj'):
            os.remove(savePath + "River/" + cat_Name + '.prj')
        if os.path.exists(savePath + "River/" + cat_Name + '.shp'):
            os.remove(savePath + "River/" + cat_Name + '.shp')
        if os.path.exists(savePath + "River/" + cat_Name + '.shx'):
            os.remove(savePath + "River/" + cat_Name + '.shx')


if __name__ == "__main__":
    ts = time()
    
    warnings.filterwarnings('ignore')
    plt.style.use('ggplot')
    # 1. Instantiate grid from raster
    filePath = 'J:/Hydrosheds_cat/'
    savePath = 'J:/WFIUH/'
    station_CN_raw = pd.read_excel(filePath + "/hydroshed_12_TP_outlets_final.xlsx", sheet_name="hydroshed_12_TP_outlets_final")
    station_CN = station_CN_raw.drop_duplicates(subset=['catID'], keep="first")
    station_CN = station_CN.reset_index(drop=True)
    resolution = 90

    for i in range(0, len(station_CN["catID"])):
        cat_Name = str(station_CN["catID"][i])
        x = station_CN["lon"][i]
        y = station_CN["lat"][i]
        calculate_WFIUH(cat_Name, filePath, savePath, x, y)
        continue
    
    # attributes = pd.DataFrame({"cat_Name": station_CN["Station_name"], "acc_density": acc_density})
    # attributes.to_excel(filePath + "acc_density.xlsx")
