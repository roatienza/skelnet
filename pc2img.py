
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

PT_PATH = "dataset/point/skel"

def pc2img():
    path = PT_PATH
    files = list_files(path)
    for f in files:
        pix_file = os.path.join(path, f)
        image = np.zeros((256,256), dtype=np.uint8)
        pix_data = read_points(pix_file)
        for p in pix_data:
            x = min(round(p[0]), 255)
            y = min(round(p[1]), 255)
            image[x][y] = 255

        impath = os.path.join("images", f + ".png")
        print("Saving ... ", impath)
        imsave(impath, image, cmap='gray')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # help_ = "pix file"
    # parser.add_argument("--pix_file", default='coords_apple-1-full.pts', help=help_)
    args = parser.parse_args()
    pc2img()
