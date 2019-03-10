"""Model trainer

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from keras.utils import plot_model
from keras.optimizers import Adam


import numpy as np
import argparse
import os
from model import build_generator
from skimage.io import imsave
from utils import list_files, read_gray
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils.multi_gpu_utils import multi_gpu_model



PX_PATH = "dataset/pixel/test"
PR_PATH = "dataset/pixel/prediction"
AP_PATH = "dataset/pixel/augtrain"
AS_PATH = "dataset/pixel/augskel"

def predict_pix(model):
    files = list_files(PX_PATH)
    pix = []
    for f in files:
        pix_file = os.path.join(PX_PATH, f)
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
        print("Max: ", np.amax(pix))
        out_pix = model.predict(pix)
        out_pix[out_pix<0.2] = 0.0
        out_pix[out_pix>0.0] = 1.0
        out_pix = np.squeeze(out_pix) * 255.0
        out_pix = out_pix.astype(np.uint8)
        out_pix = np.expand_dims(out_pix, axis=2)
        out_pix = np.concatenate((out_pix, out_pix, out_pix), axis=2)
        print(out_pix.shape)
        path = os.path.join(PR_PATH, files[i])
        print("Saving ... ", path)
        imsave(path, out_pix)


def augment(input_pix, output_pix):
    # we create two instances with the same arguments
    args = dict(rotation_range=180,
                width_shift_range=0.1,
                height_shift_range=0.05,
                horizontal_flip=True,
                vertical_flip=True,
                shear_range=30,
                zoom_range=0.2)

    datagen = ImageDataGenerator(**args)
    input_gen = []
    output_gen = []
    print("input shape: ", input_pix.shape)
    print("output shape: ", output_pix.shape)
    print("Augmenting data...")
    for i in range(16):
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
    lr = 1e-3
    if epoch > 100:
        lr = 1e-4
    print('Learning rate: ', lr)
    return lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load weights"
    parser.add_argument("--weights",
                        default=None,
                        help=help_)
    help_ = "Train"
    parser.add_argument("--train",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Batch size"
    parser.add_argument("--batch_size", type=int, default=8, help=help_)

    help_ = "Number of GPUs (default is 1)"
    parser.add_argument("--gpus", type=int, default=1, help=help_)

    args = parser.parse_args()

    infile = "in_pix.npy"
    outfile = "out_pix.npy"
    print("Loading ... ", infile) 
    input_pix = np.load(infile)
    print("Loading ... ", outfile) 
    output_pix = np.load(outfile)

    print("batch size: ", args.batch_size)
    if args.train:
        input_pix, output_pix = augment(input_pix, output_pix)
    print("input shape: ", input_pix.shape)
    print("output shape: ", output_pix.shape)
    # input image dimensions.
    input_shape = input_pix.shape[1:]
    output_shape = output_pix.shape[1:]

    # normalize data.
    input_pix = input_pix.astype('float32') / 255
    output_pix = output_pix.astype('float32') / 255


    # input_shape = (256, 256, 1)
    # output_shape = (256, 256, 1)
    model_ = build_generator(input_shape, output_shape)
    model_.summary()
    #plot_model(model, to_file='skelpix.png', show_shapes=True)


    # prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'weights')
    # model_name = 'skelnet_pix_model.{epoch:03d}.h5' 
    model_name = 'skelnet_pix_model.h5' 
    if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 verbose=1,
                                 save_weights_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    callbacks = [checkpoint, lr_scheduler]

    if args.weights is not None:
        print("Loading weights ...", args.weights)
        model_.load_weights(args.weights)
    if not args.train:
        predict_pix(model_)
    else:
        if args.gpus <= 1:
            pass
            model = model_
        else:
            model = multi_gpu_model(model_, gpus=args.gpus)


        optimizer = Adam(lr=1e-3)
        # loss = ['binary_crossentropy', 'mae']
        # loss_weights = [1., 1.]
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        # train the model with input images and labels
        model.fit(input_pix,
                  output_pix,
                  epochs=200,
                  batch_size=args.batch_size,
                  callbacks=callbacks)

        model_.save_weights("weights/weights_pix.h5")
        if args.gpus > 1:
            model.save_weights("weights/weights_gpus_pix.h5")
