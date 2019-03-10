"""Model builder

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Conv2DTranspose, Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import Model
from keras.models import load_model
from keras.layers.merge import concatenate
from keras.utils import plot_model

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import numpy as np
import argparse


def encoder_layer(inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  activation='relu',
                  instance_norm=True):

    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation(activation)(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    # x = Dropout(0.2)(x)
    return x


def decoder_layer(inputs,
                  paired_inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  activation='relu',
                  instance_norm=True):

    conv = Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation(activation)(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    x = concatenate([x, paired_inputs])
    # x = Dropout(0.2)(x)
    return x


def build_generator(input_shape,
                    output_shape=None,
                    kernel_size=3,
                    name=None):

    inputs = Input(shape=input_shape)
    channels = int(output_shape[-1])
    e1 = encoder_layer(inputs,
                       32,
                       kernel_size=kernel_size,
                       strides=1)
    # 256 x 32
    e2 = encoder_layer(e1,
                       64,
                       kernel_size=kernel_size)
    # 128 x 64
    e3 = encoder_layer(e2,
                       128,
                       kernel_size=kernel_size)
    # 64 x 128
    e4 = encoder_layer(e3,
                       256,
                       kernel_size=kernel_size)
    # 32 x 256
    e5 = encoder_layer(e4,
                       512,
                       kernel_size=kernel_size)
    # 16 x 512
    e6 = encoder_layer(e5,
                       1024,
                       kernel_size=kernel_size)
    # 8 x 1024
    e7 = encoder_layer(e6,
                       2048,
                       kernel_size=kernel_size)
    # 4 x 2048

    d0 = decoder_layer(e7,
                       e6,
                       1024,
                       kernel_size=kernel_size)
    # 8 x 1024+1024 

    d1 = decoder_layer(d0,
                       e5,
                       512,
                       kernel_size=kernel_size)
    # 16 x 512+512 
    d2 = decoder_layer(d1,
                       e4,
                       256,
                       kernel_size=kernel_size)
    # 32 x 256+256
    d3 = decoder_layer(d2,
                       e3,
                       128,
                       kernel_size=kernel_size)
    # 64 x 128+128
    d4 = decoder_layer(d3,
                       e2,
                       64,
                       kernel_size=kernel_size)
    # 128 x 64+64
    d5 = decoder_layer(d4,
                       e1,
                       32,
                       kernel_size=kernel_size)
    # 256 x 32+32
    outputs = Conv2DTranspose(channels,
                              kernel_size=kernel_size,
                              strides=1,
                              activation='sigmoid',
                              padding='same')(d5)
    # 256x256x1

    unet = Model(inputs, outputs, name=name)

    return unet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Train cifar10 colorization"
    parser.add_argument("-c",
                        "--cifar10",
                        action='store_true',
                        help=help_)
    args = parser.parse_args()
    input_shape = (256, 256, 1)
    output_shape = (256, 256, 1)
    unet = build_unet(input_shape, output_shape)
    unet.summary()
    plot_model(unet, to_file='skelpix.png', show_shapes=True)



