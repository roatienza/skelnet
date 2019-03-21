import numpy as np
from numpy.linalg import norm
import matplotlib.pylab  as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import math
import skimage
from skimage.io import imread
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

def mae_bc(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1) + K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def rotate(inputs, outputs, ntimes=8):
    args = dict(rotation_range=355)
    print("Rotating...")
    return transform(inputs, outputs, ntimes=ntimes, args=args)

def translate(inputs, outputs, ntimes=8):
    args = dict(width_shift_range=0.2,
                height_shift_range=0.2)
    print("Translating...")
    return transform(inputs, outputs, ntimes=ntimes, args=args)

def flip(inputs, outputs, ntimes=8):
    args = dict(vertical_flip=True,
                horizontal_flip=True)
    print("Flipping...")
    return transform(inputs, outputs, ntimes=2, args=args)

def scale(inputs, outputs, ntimes=8):
    args = dict(zoom_range=[0.5, 0.9])
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
    x1, y1 = rotate(inputs, outputs, ntimes=ntimes)
    x2, y2 = translate(inputs, outputs, ntimes=ntimes)
    x3, y3 = scale(inputs, outputs, ntimes=ntimes)
    x4, y4 = flip(inputs, outputs, ntimes=ntimes)
    x = np.concatenate((x1, x2, x3, x4), axis=0)
    y = np.concatenate((y1, y2, y3, y4), axis=0)
    return x, y


def augment_(inputs, outputs, shift=False, ispts=False):
    # we create two instances with the same arguments
    ntimes = 1
    print("input shape: ", inputs.shape)
    print("output shape: ", outputs.shape)
    if shift:
        args = dict(width_shift_range=0.1,
                    height_shift_range=0.1)
        ntimes = 1
        print("Augmenting data by shifting...")
    else:
        if ispts:
            args = dict(rotation_range=30,
                        vertical_flip=True,
                        zoom_range=[0.8, 1.])
        else:
            args = dict(rotation_range=30,
                        horizontal_flip=True,
                        zoom_range=[0.8, 1.])
        print("Augmenting data...")

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

    print(input_gen.shape)
    print(output_gen.shape)

    inputs = np.concatenate((inputs, input_gen), axis=0)
    outputs = np.concatenate((outputs, output_gen), axis=0)
    print("Augmented input shape: ", inputs.shape)
    print("Augmented output shape: ", outputs.shape)
    return inputs, outputs

def read_gray(f):
    # im = skimage.img_as_float(imread(f))
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

def plot_2d_point_cloud(pts,
                        sks,
                        show=True,
                        show_axis=False,
                        in_u_sphere=False,
                        marker='.',
                        s=2,
                        alpha=.8,
                        figsize=(5, 5),
                        axis=None,
                        title=None,
                        filename=None,
                        colorize=None,
                        *args,
                        **kwargs):

    x = pts[:,0] 
    y = pts[:,1] 
    a = sks[:,0] 
    b = sks[:,1] 
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)        
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    if colorize is not None:
        cm = plt.get_cmap(colorize)
        col = [cm(float(i)/(x.shape[0])) for i in range(x.shape[0])]
        sc = ax.scatter(x, y, marker=marker, s=s, alpha=alpha, c=col, *args, **kwargs)
    else:
        sc = ax.scatter(x, y, marker=marker, s=s, alpha=alpha, *args, **kwargs)
        sc = ax.scatter(a, b, marker=marker, s=s, alpha=alpha, c="red", *args, **kwargs)

    # if not show_axis:
    #    plt.axis('off')

    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
    
    plt.close('all')
    return fig



def plot_3d_point_cloud(x,
                        y,
                        z,
                        show=True,
                        show_axis=False,
                        in_u_sphere=False,
                        marker='o',
                        s=10,
                        alpha=.8,
                        figsize=(5, 5),
                        elev=10,
                        azim=240,
                        axis=None,
                        title=None,
                        filename=None,
                        colorize=None,
                        *args,
                        **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')        
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    if colorize is not None:
        cm = plt.get_cmap(colorize)
        col = [cm(float(i)/(x.shape[0])) for i in range(x.shape[0])]
        sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, c=col, *args, **kwargs)
    else:
        sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)

    # sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        # Multiply with 0.7 to squeeze free-space.
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])  
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        #plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
    
    plt.close('all')
    return fig


