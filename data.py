
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

import matplotlib.pyplot as plt

from utils import read_points, plot_3d_point_cloud, plot_2d_point_cloud
from utils import list_files, read_gray

#53941
#12270
#dataset/point/full/coords_rat-09-full.pts
#dataset/point/skel/coords_pocket-2-skel.pts

PX_PATH = "dataset/pixel/train"
SK_PATH = "dataset/pixel/skel"

def get_in_pix(filename="in_pix.npy"):
    files = list_files(PX_PATH)
    pix = []
    for f in files:
        pix_file = os.path.join(PX_PATH, f)
        pix_data =  read_gray(pix_file)
        print(pix_file)
        pix.append(pix_data)

    pix = np.array(pix)
    # pix = np.reshape(pix, [pix.shape,-1])
    print("Shape: ", pix.shape)
    pix = np.expand_dims(pix, axis=3)
    print("Final shape: ", pix.shape)
    print("Min: ", np.amin(pix))
    print("Max: ", np.amax(pix))
    print("Saving to ", filename) 
    np.save(filename, pix)
    return pix


def get_out_pix(filename="out_pix.npy"):
    files = list_files(SK_PATH)
    pix = []
    for f in files:
        pix_file = os.path.join(SK_PATH, f)
        pix_data =  read_gray(pix_file)
        print(pix_file)
        pix.append(pix_data)

    pix = np.array(pix)
    pix = np.mean(pix, axis=3)
    pix = pix.astype(np.uint8)
    print("Shape: ", pix.shape)
    print("Uniques: ", np.unique(pix))
    pix = np.expand_dims(pix, axis=3)
    print("Final shape: ", pix.shape)
    print("Min: ", np.amin(pix))
    print("Max: ", np.amax(pix))
    print("Saving to ", filename) 
    np.save(filename, pix)
    return pix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # help_ = "pix file"
    # parser.add_argument("--pix_file", default='coords_apple-1-full.pts', help=help_)
    args = parser.parse_args()

    # pix = get_in_pix()
    pix = get_out_pix()
