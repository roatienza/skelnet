'''Utils for processing images and point clouds

'''


import numpy as np
import os
from skimage.io import imread
from keras.preprocessing.image import ImageDataGenerator

def rotate(inputs, outputs, ntimes=8):
    args = dict(rotation_range=355)
    print("Rotating...")
    return transform(inputs, outputs, ntimes=ntimes, args=args)

def translate(inputs, outputs, ntimes=8):
    args = dict(width_shift_range=0.2,
                height_shift_range=0.2)
    print("Translating...")
    return transform(inputs, outputs, ntimes=ntimes, args=args)

def flip(inputs, outputs, ntimes=1):
    args = dict(horizontal_flip=True)
    print("Flipping...")
    return transform(inputs, outputs, ntimes=ntimes, args=args)

def scale(inputs, outputs, ntimes=8):
    args = dict(zoom_range=[0.6, 0.9])
    print("Scaling...")
    return transform(inputs, outputs, ntimes=ntimes, args=args)

def transform(inputs, outputs, ntimes=8, args=None):
    datagen = ImageDataGenerator(**args)
    input_gen = []
    output_gen = []
    for i in range(ntimes):
        for j in range(len(inputs)):
            inp = inputs[j]
            out = outputs[j]
            trans = datagen.get_random_transform(inp.shape)
            inp = datagen.apply_transform(inp, trans)
            out = datagen.apply_transform(out, trans)
            input_gen.append(inp)
            output_gen.append(out)

    input_gen = np.array(input_gen)
    output_gen = np.array(output_gen)

    return input_gen, output_gen


def augment(inputs, outputs, ispts=False, ntimes=8):
    print("Augmenting for %d times..." % ntimes)
    x1, y1 = rotate(inputs, outputs, ntimes=ntimes)
    x2, y2 = translate(inputs, outputs, ntimes=ntimes)
    x3, y3 = scale(inputs, outputs, ntimes=ntimes)
    x4, y4 = flip(inputs, outputs, ntimes=1)
    x = np.concatenate((x1, x2, x3, x4), axis=0)
    y = np.concatenate((y1, y2, y3, x4), axis=0)
    return x, y


def read_gray(f):
    im = imread(f)
    return im


def list_files(dir_):
    files = []
    files.extend([f for f in sorted(os.listdir(dir_)) ])
    return files


def read_points(f):
    fh = open(f, "r")
    lines = fh.readlines()

    pts = []
    for line in lines:
        line = line.strip()
        pt = [float(x) for x in line.split(" ")]
        pts.append(pt)

    fh.close()
    return pts
