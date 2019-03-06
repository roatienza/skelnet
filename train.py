"""Model trainer

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils import plot_model
from keras.optimizers import Adam


import numpy as np
import argparse
from model import build_unet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Train cifar10 colorization"
    parser.add_argument("-c",
                        "--cifar10",
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
    model = build_unet(input_shape, output_shape)
    model.summary()
    plot_model(model, to_file='skelpix.png', show_shapes=True)
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    # train the model with input images and labels
    model.fit(input_pix,
              output_pix,
              epochs=20,
              batch_size=8)
    model.save_weights("weights_pix.h5")
