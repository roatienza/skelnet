
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow as tf
# tf.enable_eager_execution()

import numpy as np
import argparse
import sys

import os
import datetime
import scipy.misc
import json
from skimage.io import imsave
from scipy.io import loadmat
from math import ceil

import matplotlib.pyplot as plt

from utils import read_points, plot_3d_point_cloud, plot_2d_point_cloud
from utils import list_files, read_gray

PT_PATH = "dataset/bezier/test"
PX_PATH = "dataset/bezier/train"
MAT_PATH = "dataset/bezier/skel/mat"
CSV_PATH = "dataset/bezier/skel/csv"

def get_in_pix(filename="in_bez.npy", ispix=True, istest=False):
    path = PX_PATH
    files = list_files(path)
    pix = []
    pmax = 0
    pmin = 255
    for f in files:
        pix_file = os.path.join(path, f)
        print(pix_file)
        pix_data =  read_gray(pix_file)
        pix.append(pix_data)

    pix = np.array(pix)
    print("Shape: ", pix.shape)
    print("PMin: ", pmin)
    print("PMax: ", pmax)
    if not istest:
        pix = np.expand_dims(pix, axis=3)
    print("Final shape: ", pix.shape)
    print("Min: ", np.amin(pix))
    print("Max: ", np.amax(pix))
    if not istest:
        print("Saving to ", filename) 
        np.save(filename, pix)
    return pix


# (630, 4)
#(1219, 630, 4)
#Max params:  630
#Min 0:  -391.4641281767936
#Max 0:  765.293070094312
#Min 1:  -288.52191910103204
#Max 1:  592.0539507793941
#Min 2:  -102.74838321431356
#Max 2:  174.9191383591248
#Min 3:  0.0
#Max 3:  1.0
#Saving to  out_bez.npy


def get_out_pix(filename="out_bez.npy"):
    files = list_files(MAT_PATH)
    pix = []
    max_pt = 0
    for f in files:
        pix_file = os.path.join(MAT_PATH, f)
        pix_data = loadmat(pix_file)
        pix_data = pix_data['bzpointsArray']
        print(pix_file)
        print(pix_data.shape)
        # pix_data = np.expand_dims(pix_data, axis=2)
        # pix_data[:-1] = 1.0
        ones = np.ones((pix_data.shape[0],1)).astype(np.uint8)
        print(pix_data.shape)
        if pix_data.shape[0] > max_pt:
            max_pt = pix_data.shape[0]
        pix_data = np.append(pix_data, ones, axis=-1)
        pad = np.zeros((630, 4))
        pad[:pix_data.shape[0], :pix_data.shape[1]] = pix_data
        print(pad)
        print(pad.shape)
        pix.append(pad)


    pix = np.array(pix)
    print(pix.shape)
    print("Max params: ", max_pt)
    print("Min 0: ", np.amin(pix[:,:,0]))
    print("Max 0: ", np.amax(pix[:,:,0]))
    print("Min 1: ", np.amin(pix[:,:,1]))
    print("Max 1: ", np.amax(pix[:,:,1]))
    print("Min 2: ", np.amin(pix[:,:,2]))
    print("Max 2: ", np.amax(pix[:,:,2]))
    print("Min 3: ", np.amin(pix[:,:,3]))
    print("Max 3: ", np.amax(pix[:,:,3]))
    print("--------------")
    c = ceil(max(abs(np.amin(pix[:,:,0])),np.amax(pix[:,:,0])))
    pix[:,:,0] /= c
    print("ceil 0:", c)
    c = ceil(max(abs(np.amin(pix[:,:,1])),np.amax(pix[:,:,1])))
    pix[:,:,1] /= c
    print("ceil 1:", c)
    c = ceil(max(abs(np.amin(pix[:,:,2])),np.amax(pix[:,:,2])))
    pix[:,:,2] /= c
    print("ceil 2:", c)
    print("Min 0: ", np.amin(pix[:,:,0]))
    print("Max 0: ", np.amax(pix[:,:,0]))
    print("Min 1: ", np.amin(pix[:,:,1]))
    print("Max 1: ", np.amax(pix[:,:,1]))
    print("Min 2: ", np.amin(pix[:,:,2]))
    print("Max 2: ", np.amax(pix[:,:,2]))
    print("Min 3: ", np.amin(pix[:,:,3]))
    print("Max 3: ", np.amax(pix[:,:,3]))
    print("Saving to ", filename) 
    np.save(filename, pix)
    return pix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # pix = get_in_pix()
    pix = get_out_pix()
