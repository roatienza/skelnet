
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

from utils import read_points
from utils import list_files, read_gray

#53941
#12270
#dataset/point/full/coords_rat-09-full.pts
#dataset/point/skel/coords_pocket-2-skel.pts

PT_PATH = "dataset/point/test"
ROOT_PATH = "dataset/point/root"
PR_PATH = "dataset/point/pred"

def img2pt():
    path = ROOT_PATH
    files = list_files(path)
    pix = []
    for f in files:
        pix_file = os.path.join(path, f)
        print(pix_file)
        filename = pix_file
        filename = filename.replace(".png", "")
        filename = filename.replace("full", "skel")
        filename = filename.replace("root", "pred")
        print(filename)
        pix_data =  read_gray(pix_file)
        with open(filename, "w+") as  fh:
            for x in range(pix_data.shape[0]):
                for y in range(pix_data.shape[0]):
                    if pix_data[x][y]>0:
                        fh.write("%d %d\n" % (x, y))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # help_ = "pix file"
    # parser.add_argument("--pix_file", default='coords_apple-1-full.pts', help=help_)
    args = parser.parse_args()
    img2pt()

