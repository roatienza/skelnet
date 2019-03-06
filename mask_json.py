
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
    fig = plt.figure(figsize=(8, 8))

    f = "apple-1.png"
    # if True:
    for f in files:
        sk_file = f.replace("full", "skel")
        pt_file = os.path.join(PT_PATH, f)
        pt_size = os.stat(pt_file).st_size
        sk_file = os.path.join(SK_PATH, sk_file)
        sk_size = os.stat(sk_file).st_size
        
        pt_data =  read_gray(pt_file)

        sk_data =  read_gray(sk_file)
        x = []
        y = []
        for i in range(sk_data.shape[0]):
            for j in range(sk_data.shape[1]):
                m = sk_data[i, j, 0] + sk_data[i, j, 1] + sk_data[i, j, 2]
                if m > 0:
                    x.append(j)
                    y.append(i)

        key = f + str(pt_size)
        points = {}
        for i in range(len(x)):
            points[i] = {
                            "shape_attributes": {
                                "name": "point",
                                "cx": x[i],
                                "cy": y[i]
                            },
                            "region_attributes": {}
                        }
        value = {   "base64_img_data" : "",
                    "file_attributes" : {},
                    "filename" : f,
                    "fileref" : "",
                    "size" : str(pt_size),
                    "regions": points
                }
        json_str[key] = value
        print(pt_file)

    with open("via.json", 'w') as outfile:
        json.dump(json_str, outfile, indent=4, sort_keys=True)

