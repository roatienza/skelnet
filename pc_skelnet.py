
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
from utils import read_points, plot_3d_point_cloud, plot_2d_point_cloud
from utils import list_files


PC_PATH = "dataset/point/full"
SK_PATH = "dataset/point/skel"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "pc file"
    parser.add_argument("--pc_file", default='coords_apple-1-full.pts', help=help_)
    args = parser.parse_args()

    files = list_files(PC_PATH)
    mfull = [0, 0]
    sfull = [0, 0]
    for f in files:
        sk_file = f.replace("full", "skel")
        pc_file = os.path.join(PC_PATH, f)
        print(pc_file)
        pts = np.array(read_points(pc_file))
        sk_file = os.path.join(SK_PATH, sk_file)
        sks = np.array(read_points(sk_file))
        plot_2d_point_cloud(pts, sks)


    exit(0)

    print(files)
    files = list_files(SK_PATH)
    print(files)

    sk_file = args.pc_file.replace("full", "skel")
    pc_file = os.path.join(PC_PATH, args.pc_file)
    sk_file = os.path.join(SK_PATH, sk_file)
    print(pc_file)
    print(sk_file)
    pts = np.array(read_points(pc_file))
    max_x = np.amax(pts[:,0])
    max_y = np.amax(pts[:,1])
    print(max_x)
    print(max_y)
    sks = np.array(read_points(sk_file))
    plot_2d_point_cloud(pts, sks)


