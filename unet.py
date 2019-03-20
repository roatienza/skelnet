"""Model trainer

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import argparse
import os
from unet_model import build_generator
from skimage.io import imsave
from utils import list_files, read_gray, augment, mae_bc
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam, RMSprop
from keras.models import Model
from keras.layers import Input
import datetime


PT_PATH = "dataset/pixel/train"
PX_PATH = "dataset/pixel/test"
PR_PATH = "dataset/pixel/root"
EPOCHS = 100

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

    pix = np.array(pix)
    print("Shape: ", pix.shape)
    input_pix = np.expand_dims(pix, axis=3)
    input_pix = input_pix / 255.0
    print("Final shape: ", pix.shape)


    for i in range(input_pix.shape[0]):
        pix = input_pix[i]
        pix = np.expand_dims(pix, axis=0)
        out_pix = unet.predict([pix, pix, pix, pix])
        print("Max: ", np.amax(pix))
        out_pix[out_pix>=0.2] = 1.0
        out_pix[out_pix<0.1] = 0.0
        out_pix = np.squeeze(out_pix) * 255.0
        out_pix = out_pix.astype(np.uint8)
        print(out_pix.shape)
        path = os.path.join(PR_PATH, files[i])
        if ispt:
            path = path.replace("pixel", "point")
        print("Saving ... ", path)
        if ispt:
            imsave(path, out_pix, cmap='gray')
        else:
            out_pix = np.expand_dims(out_pix, axis=2)
            out_pix = np.concatenate((out_pix, out_pix, out_pix), axis=2)
            imsave(path, out_pix)

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 40:
        lr = 1e-4
    print('Learning rate: ', lr)
    return lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Model weights"
    parser.add_argument("--weights",
                        default=None,
                        help=help_)
    help_ = "Train model"
    parser.add_argument("--train",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Plot model"
    parser.add_argument("--plot",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Skip connection"
    parser.add_argument("--skip",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Pixel mode instead of point mode"
    parser.add_argument("--pix",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Batch size"
    parser.add_argument("--batch_size", type=int, default=8, help=help_)

    help_ = "ntimes to augment data"
    parser.add_argument("--ntimes", type=int, default=8, help=help_)

    help_ = "Number of GPUs (default is 1)"
    parser.add_argument("--gpus", type=int, default=1, help=help_)

    args = parser.parse_args()

    if args.pix:
        infile = "in_pix.npy"
        outfile = "out_pix.npy"
    else:
        infile = "in_pts.npy"
        outfile = "out_pts.npy"
    print("Loading ... ", infile) 
    input_pix = np.load(infile)
    print("Loading ... ", outfile) 
    output_pix = np.load(outfile)

    print("batch size: ", args.batch_size)
    input_shape = input_pix.shape[1:]
    output_shape = output_pix.shape[1:]

    if args.skip:
        print("Building PSP UNet with skip connection")
    else:
        print("Building PSP UNet")
    unet = build_generator(input_shape, output_shape, kernel_size=3, skip=args.skip)
    unet.summary()

    if args.plot:
        from keras.utils import plot_model
        plot_model(unet, to_file='unet.png', show_shapes=True)

    if args.weights is not None:
        print("Loading unet weights ...", args.weights)
        unet.load_weights(args.weights)

    if not args.train:
        if args.pix:
            predict_pix(unet, ispt=False)
        else:
            predict_pix(unet, ispt=True)
    else:
        optimizer = Adam(lr=1e-3)
        unet.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'weights')
        if args.pix:
            model_name = 'skelnet_pix_unet_model.h5' 
            if args.skip:
                model_name = 'skelnet_pix_unet_skip_model.h5' 
        else:
            model_name = 'skelnet_pts_unet_model.h5' 
            if args.skip:
                model_name = 'skelnet_pts_unet_skip_model.h5' 

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
        for i in range(4):
            x, y = augment(input_pix, output_pix, ntimes=args.ntimes)
            x = np.concatenate((input_pix, x), axis=0)
            y = np.concatenate((output_pix, y), axis=0)
            print("Augmented input shape: ", x.shape)
            print("Augmented output shape: ", y.shape)
            x = x.astype('float32') / 255
            y = y.astype('float32') / 255
            inputs = [x, x, x]
            unet.fit(inputs,
                     y,
                     validation_data=(xval, yval),
                     epochs=60,
                     batch_size=args.batch_size,
                     callbacks=callbacks)
            x = None
            y = None
            del x
            del y
