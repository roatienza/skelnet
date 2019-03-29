"""Model trainer

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import argparse
import os
from pt_model import build_generator
from skimage.io import imsave
from utils import list_files, read_gray, augment
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam, RMSprop
from keras.models import Model
from keras.layers import Input
import datetime

from other_utils import test_generator, display_images


PT_PATH = "dataset/pixel/train"
PX_PATH = "dataset/pixel/test"
PR_PATH = "dataset/pixel/root"
EPOCHS = 140

def predict_pix(model, path=PX_PATH, ispt=False):
    if ispt:
        path = path.replace("pixel", "point")
        path = path.replace("test", "test_img")
    files = list_files(path)
    pix = []
    for f in files:
        pix_file = os.path.join(path, f)
        pix_data =  read_gray(pix_file)
        print(pix_file)
        pix.append(pix_data)

    global thresh
    pix = np.array(pix)
    print("Shape: ", pix.shape)
    input_pix = np.expand_dims(pix, axis=3)
    input_pix = input_pix / 255.0
    print("Final shape: ", pix.shape)

    maxpts = 1024 * 12

    pts = []
    for i in range(input_pix.shape[0]):
        pix = input_pix[i]
        pix = np.expand_dims(pix, axis=0)
        out_pix = generator.predict([pix, pix, pix])
        print("Max: ", np.amax(pix))
        out_pix[out_pix>=thresh] = 1.0
        out_pix[out_pix<thresh] = 0.0
        out_pix = np.squeeze(out_pix) * 255.0
        out_pix = out_pix.astype(np.uint8)
        print(out_pix.shape)
        path = os.path.join(PR_PATH, files[i])
        if ispt:
            path = path.replace("pixel", "point")
        print("Saving ... ", path)
        if ispt:
            imsave(path, out_pix, cmap='gray')

            pt = np.zeros((maxpts, 3))
            j = 0
            for x in range(out_pix.shape[0]):
                for y in range(out_pix.shape[1]):
                    if out_pix[x][y]>0:
                        pt[j] = (x, y, 0)
                        j += 1
                        if j >= (maxpts - 1):
                            j = maxpts - 1
            pts.append(pt)
        else:
            out_pix = np.expand_dims(out_pix, axis=2)
            out_pix = np.concatenate((out_pix, out_pix, out_pix), axis=2)
            imsave(path, out_pix)

    pts = np.array(pts)
    pts = pts.astype(np.uint8)
    print("Skel test shape:", pts.shape)
    print("Skel test max:", np.amax(pts))
    print("Skel test min:", np.amin(pts))
    print("Skel test dtype:", pts.dtype)
    filename = "npy/test_pc.npy"
    print("Saving to ", filename) 
    np.save(filename, pts)

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 100:
        lr = 1e-5
    elif epoch > 60:
        lr = 1e-4
    print('Learning rate: ', lr)
    return lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Gen weights"
    parser.add_argument("--gen",
                        default=None,
                        help=help_)
    help_ = "Train"
    parser.add_argument("--train",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Plot model"
    parser.add_argument("--plot",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Aug"
    parser.add_argument("--aug",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Pix"
    parser.add_argument("--pix",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "No dropout"
    parser.add_argument("--nodropout",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Batch size"
    parser.add_argument("--batch_size", type=int, default=8, help=help_)

    help_ = "Kernel size"
    parser.add_argument("--kernel_size", type=int, default=3, help=help_)

    help_ = "ntimes"
    parser.add_argument("--ntimes", type=int, default=8, help=help_)

    args = parser.parse_args()
    thresh = 0.5

    if args.pix:
        infile = "npy/in_pix.npy"
        outfile = "npy/out_pix.npy"
    else:
        infile = "npy/in_pts.npy"
        outfile = "npy/out_pts.npy"
    print("Loading ... ", infile) 
    input_pix = np.load(infile)
    print("input shape: ", input_pix.shape)
    print("Loading ... ", outfile) 
    output_pix = np.load(outfile)
    print("output shape: ", output_pix.shape)

    print("batch size: ", args.batch_size)
    input_shape = input_pix.shape[1:]
    output_shape = output_pix.shape[1:]

    generator = build_generator(input_shape, output_shape, kernel_size=args.kernel_size, nodropout=args.nodropout)

    if args.plot:
        from keras.utils import plot_model
        plot_model(generator, to_file='generator.png', show_shapes=True)

    if args.gen is not None:
        print("Loading generator weights ...", args.gen)
        generator.load_weights(args.gen)

    if not args.train:
        if args.pix:
            predict_pix(generator, ispt=False)
        else:
            predict_pix(generator, ispt=True)
    else:
        optimizer = Adam(lr=1e-3)
        generator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'weights')
        if args.pix:
            model_name = 'skelnet_pix_model.h5' 
        else:
            if args.nodropout:
                model_name = 'skelnet_pt_nodropout_model.h5' 
            else:
                model_name = 'skelnet_pt_model.h5' 
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        # prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     verbose=1,
                                     save_weights_only=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        callbacks = [checkpoint, lr_scheduler]

        # train the model with input images and labels
        xval = input_pix.astype('float32') / 255
        xval = [xval, xval, xval]
        yval = output_pix.astype('float32') / 255

        x, y = augment(input_pix, output_pix, ntimes=args.ntimes)
        x = np.concatenate((input_pix, x), axis=0)
        y = np.concatenate((output_pix, y), axis=0)
        print("Augmented input shape: ", x.shape)
        print("Augmented output shape: ", y.shape)
        x = x.astype('float32') / 255
        y = y.astype('float32') / 255
        inputs = [x, x, x]
        generator.fit(inputs,
                      y,
                      epochs=140,
                      validation_data=(xval, yval),
                      batch_size=args.batch_size,
                      callbacks=callbacks)

