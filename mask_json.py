
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

from utils import read_points, plot_3d_point_cloud, plot_2d_point_cloud
from utils import list_files, read_gray

#53941
#12270
#dataset/point/full/coords_rat-09-full.pts
#dataset/point/skel/coords_pocket-2-skel.pts

PT_PATH = "dataset/pixel/train"
SK_PATH = "dataset/pixel/skel"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # help_ = "pix file"
    # parser.add_argument("--pix_file", default='coords_apple-1-full.pts', help=help_)
    args = parser.parse_args()

    files = list_files(PT_PATH)
    pts = np.array([])
    sks = np.array([])
    maxplen = 0
    maxslen = 0
    pmin = 255
    pmax = 0
    smin = 255
    smax = 0

    json_str = {}
    for f in files:
        sk_file = f.replace("full", "skel")
        pt_file = os.path.join(PT_PATH, f)
        pt_size = os.stat(pt_file).st_size
        # pt = np.array(read_points(pt_file))
        sk_file = os.path.join(SK_PATH, sk_file)
        sk_size = os.stat(sk_file).st_size
        # sk = np.array(read_points(sk_file))
        # pts = np.append(pts, pt, axis=0)
        # sks = np.append(sks, sk, axis=0)
        #if pt.shape[0] > maxplen:
        #    p = pc_file
        #if sk.shape[0] > maxslen:
        #    s = sk_file
        #maxplen = max(maxplen, pt.shape[0])
        #maxslen = max(maxslen, sk.shape[0])
        #pmax = max(pmax, np.amax(pt))
        #pmin = min(pmin, np.amin(pt))
        #smax = max(smax, np.amax(sk))
        #smin = min(smin, np.amin(sk))
        
        # plot_2d_point_cloud(pts, sks)
        print(pt_file, pt_size)
        pt_data =  read_gray(pt_file)
        print(pt_data.shape)
        print(pt_data.dtype)
        print(pt_data)

        print(sk_file, sk_size)
        sk_data =  read_gray(sk_file)
        print(sk_data.shape)
        print(sk_data.dtype)
        print(np.unique(sk_data))
        for i in range(sk_data.shape[0]):
            for j in range(sk_data.shape[1]):
                print(sk_data[i, j, 0], end= " ")

        key = f + str(pt_size)
        value = {   "base64_img_data" : "",
                    "file_attributes" : {},
                    "filename" : f,
                    "fileref" : "",
                    "size" : str(pt_size)
                }
        json_str[key] = value
        break

    with open("via.json", 'w') as outfile:
        json.dump(json_str, outfile)

