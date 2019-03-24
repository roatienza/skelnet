"""Model trainer

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import argparse
import os
from bezier_model import build_model, build_generator, build_discriminator
from skimage.io import imsave
from utils import list_files, read_gray, augment, mae_bc
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam, RMSprop
from keras.models import Model
from keras.layers import Input
import datetime

from other_utils import test_generator, display_images


EPOCHS = 100
PX_PATH = "dataset/bezier/test"
PR_PATH = "dataset/bezier/root"


def predict_pix(model, path=PX_PATH):
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
    print(input_pix[0])
    print("Final shape: ", pix.shape)
    print("Max: ", np.amax(input_pix))
    return

    for i in range(input_pix.shape[0]):
        pix = input_pix[i]
        pix = np.expand_dims(pix, axis=0)
        pred = model.predict(pix)
        print("Min: ", np.amin(pred))
        print("Max: ", np.amax(pred))

        pred = np.squeeze(pred)
        print(pred)
        print(pred.shape)
        filename = files[i]
        filename = filename.replace("png", ".csv")
        filename = "bzskeleton_skelpoints_" + filename
        path = os.path.join(PR_PATH, filename)
        print(path)
        


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 40:
        lr *= 1e-1 
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

    help_ = "Batch size"
    parser.add_argument("--batch_size", type=int, default=8, help=help_)

    help_ = "ntimes"
    parser.add_argument("--ntimes", type=int, default=8, help=help_)

    help_ = "Number of GPUs (default is 1)"
    parser.add_argument("--gpus", type=int, default=1, help=help_)

    args = parser.parse_args()

    infile = "in_bez.npy"
    outfile = "out_bez.npy"

    print("Loading ... ", infile) 
    input_pix = np.load(infile)
    print("Loading ... ", outfile) 
    output_pix = np.load(outfile)

    print("batch size: ", args.batch_size)
    input_shape = input_pix.shape[1:]
    output_shape = output_pix.shape[1:]

    generator = build_model(input_shape, output_shape)
    generator.summary()

    if args.plot:
        from keras.utils import plot_model
        plot_model(generator, to_file='generator.png', show_shapes=True)

    if args.gen is not None:
        print("Loading generator weights ...", args.gen)
        generator.load_weights(args.gen)

    if not args.train:
        print("Min: ", np.amin(output_pix))
        print("Max: ", np.amax(output_pix))
        predict_pix(generator)
    else:
        optimizer = Adam(lr=1e-3)
        generator.compile(loss='mse',
                          optimizer=optimizer,
                          metrics=['accuracy'])

        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'weights')
        model_name = 'skelnet_bezier_model.h5' 
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
        yval = output_pix
        # yval = output_pix.astype('float32') / 255
        # x, y = augment(input_pix, output_pix, ntimes=args.ntimes)
        #x = np.concatenate((input_pix, x), axis=0)
        #y = np.concatenate((output_pix, y), axis=0)
        #print("Augmented input shape: ", x.shape)
        #print("Augmented output shape: ", y.shape)
        #x = x.astype('float32') / 255
        #y = y.astype('float32') / 255
        inputs = xval
        outputs = yval
        generator.fit(inputs,
                      outputs,
                      epochs=60,
                      validation_data=(xval, yval),
                      batch_size=args.batch_size,
                      callbacks=callbacks)
