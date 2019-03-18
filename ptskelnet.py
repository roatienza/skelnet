"""Model trainer

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import argparse
import os
from resnet_model import build_generator
from skimage.io import imsave
from utils import list_files, read_gray
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
EPOCHS = 200

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
        out_pix = generator.predict([pix, pix, pix, pix])
        print("Max: ", np.amax(pix))
        out_pix[out_pix>=0.1] = 1.0
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



def augment(input_pix, output_pix, shift=False, ispts=False):
    # we create two instances with the same arguments
    ntimes = 4
    print("input shape: ", input_pix.shape)
    print("output shape: ", output_pix.shape)
    if shift:
        args = dict(width_shift_range=0.1,
                    height_shift_range=0.1)
        ntimes = 2
        print("Augmenting data by shifting...")
    else:
        if ispts:
            args = dict(rotation_range=60,
                        vertical_flip=True,
                        zoom_range=[0.8, 1.])
        else:
            args = dict(rotation_range=60,
                        horizontal_flip=True,
                        zoom_range=[0.8, 1.])
        print("Augmenting data...")

    datagen = ImageDataGenerator(**args)
    input_gen = []
    output_gen = []
    for i in range(ntimes):
        for j in range(len(input_pix)):
            inp = input_pix[j]
            out = output_pix[j]
            trans = datagen.get_random_transform(inp.shape)
            inp = datagen.apply_transform(inp, trans)
            out = datagen.apply_transform(out, trans)
            input_gen.append(inp)
            output_gen.append(out)

    input_gen = np.array(input_gen)
    output_gen = np.array(output_gen)

    print(input_gen.shape)
    print(output_gen.shape)

    input_pix = np.concatenate((input_pix, input_gen), axis=0)
    output_pix = np.concatenate((output_pix, output_gen), axis=0)
    return input_pix, output_pix


def lr_schedule(epoch):
    #lr = 1e-3
    #if epoch > 100:
    #    lr = 0.5e-4
    #elif epoch > 40:
    #    lr = 1e-4
    #elif epoch > 20:
    #    lr = 0.5e-3
    #print('Learning rate: ', lr)

    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
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

    help_ = "Number of GPUs (default is 1)"
    parser.add_argument("--gpus", type=int, default=1, help=help_)

    args = parser.parse_args()

    infile = "in_pts.npy"
    outfile = "out_pts.npy"
    print("Loading ... ", infile) 
    input_pix = np.load(infile)
    print("Loading ... ", outfile) 
    output_pix = np.load(outfile)

    print("batch size: ", args.batch_size)
    if args.aug:
        input_pix, output_pix = augment(input_pix, output_pix)
        input_pix, output_pix = augment(input_pix, output_pix, shift=True)

    print("input shape: ", input_pix.shape)
    print("output shape: ", output_pix.shape)
    # input image dimensions.
    input_shape = input_pix.shape[1:]
    output_shape = output_pix.shape[1:]

    # normalize data.
    input_pix = input_pix.astype('float32') / 255
    output_pix = output_pix.astype('float32') / 255

    generator = build_generator(input_shape, output_shape, kernel_size=3)
    generator.summary()

    if args.plot:
        from keras.utils import plot_model
        plot_model(generator, to_file='generator.png', show_shapes=True)

    if args.gen is not None:
        print("Loading generator weights ...", args.gen)
        generator.load_weights(args.gen)

    if not args.train:
        predict_pix(generator, ispt=True)
    else:
        optimizer = Adam(lr=1e-3)
        generator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'weights')
        model_name = 'skelnet_resnet_model.h5' 
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
        inputs = [input_pix, input_pix, input_pix, input_pix]
        generator.fit(inputs,
                      output_pix,
                      epochs=EPOCHS,
                      batch_size=args.batch_size,
                      callbacks=callbacks)
