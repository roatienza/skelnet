
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

#53941
#12270
#dataset/point/full/coords_rat-09-full.pts
#dataset/point/skel/coords_pocket-2-skel.pts

PC_PATH = "dataset/point/full"
SK_PATH = "dataset/point/skel"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "pc file"
    parser.add_argument("--pc_file", default='coords_apple-1-full.pts', help=help_)
    args = parser.parse_args()

    files = list_files(PC_PATH)
    pts = np.array([])
    sks = np.array([])
    maxplen = 0
    maxslen = 0
    pmin = 255
    pmax = 0
    smin = 255
    smax = 0

    for f in files:
        sk_file = f.replace("full", "skel")
        pc_file = os.path.join(PC_PATH, f)
        pt = np.array(read_points(pc_file))
        sk_file = os.path.join(SK_PATH, sk_file)
        sk = np.array(read_points(sk_file))
        # pts = np.append(pts, pt, axis=0)
        # sks = np.append(sks, sk, axis=0)
        if pt.shape[0] > maxplen:
            p = pc_file
        if sk.shape[0] > maxslen:
            s = sk_file
        maxplen = max(maxplen, pt.shape[0])
        maxslen = max(maxslen, sk.shape[0])
        pmax = max(pmax, np.amax(pt))
        pmin = min(pmin, np.amin(pt))
        smax = max(smax, np.amax(sk))
        smin = min(smin, np.amin(sk))
        
        # plot_2d_point_cloud(pts, sks)
        print(pc_file)

    print(maxplen)
    print(maxslen)
    print(p)
    print(s)
    print(pmax, pmin)
    print(smax, smin)

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


