
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

import matplotlib.pyplot as plt

from utils import read_points, plot_3d_point_cloud, plot_2d_point_cloud
from utils import list_files, read_gray

#53941
#12270
#dataset/point/full/coords_rat-09-full.pts
#dataset/point/skel/coords_pocket-2-skel.pts

PT_PATH = "dataset/pixel/test"
PX_PATH = "dataset/pixel/train"
SK_PATH = "dataset/pixel/skel"

def get_in_pix(filename="in_pix.npy", ispix=True, isskel=False, istest=False):
    path = PX_PATH
    if istest:
        path = PT_PATH
    if isskel:
        path = SK_PATH
    if not ispix:
        path = path.replace("pixel", "point")
    files = list_files(path)
    pix = []
    pmax = 0
    pmin = 255
    maxpts = 0
    for f in files:
        pix_file = os.path.join(path, f)
        print(pix_file)
        if ispix:
            pix_data =  read_gray(pix_file)
        else:
            image = np.zeros((256,256), dtype=np.uint8)
            pix_data = read_points(pix_file)
            if len(pix_data) > maxpts:
                maxpts = len(pix_data)
            for p in pix_data:
                if p[0]>pmax:
                    pmax = p[0]
                if p[0]<pmin:
                    pmin = p[0]
                if p[1]>pmax:
                    pmax = p[1]
                if p[1]<pmin:
                    pmin = p[1]
                x = min(round(p[0]), 255)
                y = min(round(p[1]), 255)
                image[x][y] = 255
            impath = os.path.join("images", f + ".png")
            print("Saving ... ", impath)
            imsave(impath, image, cmap='gray')
            pix_data = image

        pix.append(pix_data)

    # Max pts:  12270
    print("Max pts: ", maxpts)
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

    # pix = get_in_pix(filename="in_pts.npy", ispix=False, isskel=False)
    pix = get_in_pix(filename="out_pts.npy", ispix=False, isskel=True, istest=False)
    # pix = get_out_pix()
