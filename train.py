"""Model trainer

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils import plot_model
from keras.optimizers import Adam


import numpy as np
import argparse
import os
from model import build_generator
from skimage.io import imsave
from utils import list_files, read_gray
from keras.preprocessing.image import ImageDataGenerator



PX_PATH = "dataset/pixel/test"
PR_PATH = "dataset/pixel/prediction"

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
    print("Final shape: ", pix.shape)


    for i in range(input_pix.shape[0]):
        pix = input_pix[i]
        pix = np.expand_dims(pix, axis=0)
        out_pix = model.predict(pix)
        out_pix = np.squeeze(out_pix) * 255.0
        out_pix = out_pix.astype(np.uint8)
        # out_pix[out_pix<0.5] = 0.0
        # out_pix[out_pix>0.0] = 1.0
        path = os.path.join(PR_PATH, files[i])
        print("Saving ... ", path)
        imsave(path, out_pix)

def train(input_pix, output_pix, model):
    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(input_pix, augment=True, seed=seed)
    mask_datagen.fit(output_pix, augment=True, seed=seed)

    image_generator = image_datagen.flow_from_directory(
                    'data/images',
                    class_mode=None,
                    seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
                'data/masks',
                class_mode=None,
                seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    model.fit_generator(
                train_generator,
                steps_per_epoch=2000,
                epochs=50)


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
    args = parser.parse_args()


   
    infile = "in_pix.npy"
    outfile = "out_pix.npy"
    print("Loading ... ", infile) 
    input_pix = np.load(infile)
    print("Loading ... ", outfile) 
    output_pix = np.load(outfile)

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
    model = build_generator(input_shape, output_shape)
    model.summary()
    plot_model(model, to_file='skelpix.png', show_shapes=True)
    if args.weights is not None:
        print("Loading weights ...", args.weights)
        model.load_weights(args.weights)
    if not args.train:
        predict_pix(model)
    else:
        optimizer = Adam(lr=1e-3)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        # train the model with input images and labels
        model.fit(input_pix,
                  output_pix,
                  epochs=400,
                  batch_size=8)
        model.save_weights("weights_pix.h5")