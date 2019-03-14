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
    outputs0 = encoder_layer(inputs,
                             32,
                             strides=1,
                             kernel_size=kernel_size)

    # inputs1 = Input(shape=input_shape)
    channels = int(output_shape[-1])
    e11 = encoder_layer(outputs0,
                        32,
                        kernel_size=kernel_size)
    # 128x128 x 32
    e12 = encoder_layer(e11,
                        64,
                        kernel_size=kernel_size)
    # 64x64 x 64
    e13 = encoder_layer(e12,
                        128,
                        kernel_size=kernel_size)
    # 32x32 x 128
    e14 = encoder_layer(e13,
                        256,
                        kernel_size=kernel_size)
    # 16x16 x 256
    e15 = encoder_layer(e14,
                        512,
                        kernel_size=kernel_size)
    # 8x8 x 512
    e16 = encoder_layer(e15,
                        1024,
                        kernel_size=kernel_size)
    # 4x4 x 1024

    d11 = decoder_layer(e16,
                        e15,
                        512,
                        kernel_size=kernel_size)
    # 8x8 x 512+512 
    d12 = decoder_layer(d11,
                        e14,
                        256,
                        kernel_size=kernel_size)
    # 16x16 x 256+256
    d13 = decoder_layer(d12,
                        e13,
                        128,
                        kernel_size=kernel_size)
    # 32x32 x 128+128
    d14 = decoder_layer(d13,
                        e12,
                        64,
                        kernel_size=kernel_size)
    # 64x64 x 64+64
    d15 = decoder_layer(d14,
                        e11,
                        32,
                        kernel_size=kernel_size)
    # 128x128 x 32+32

    outputs1 = Conv2DTranspose(channels,
                               kernel_size=kernel_size,
                               strides=2,
                               padding='same')(d15)
    # 256x256 x channels


    # inputs2 = Input(shape=input_shape)
    e21 = encoder_layer(outputs0,
                        32,
                        strides=4,
                        kernel_size=kernel_size)
    # 64x64 x 32
    e22 = encoder_layer(e21,
                        64,
                        strides=4,
                        kernel_size=kernel_size)
    # 16x16 x 64
    e23 = encoder_layer(e22,
                        128,
                        strides=4,
                        kernel_size=kernel_size)
    # 4x4 x 128

    d21 = decoder_layer(e23,
                        e22,
                        64,
                        strides=4,
                        kernel_size=kernel_size)
    # 16x16 x 64+64 
    d22 = decoder_layer(d21,
                        e21,
                        32,
                        strides=4,
                        kernel_size=kernel_size)
    # 64x64 x 32+32
    outputs2 = Conv2DTranspose(channels,
                               kernel_size=kernel_size,
                               strides=4,
                               padding='same')(d22)
    # 256x256 x 1

    # inputs3 = Input(shape=input_shape)
    e31 = encoder_layer(outputs0,
                        32,
                        strides=8,
                        kernel_size=kernel_size)
    # 32x32 x 32
    e32 = encoder_layer(e31,
                        64,
                        strides=8,
                        kernel_size=kernel_size)
    # 4x4 x 64

    d31 = decoder_layer(e32,
                        e31,
                        64,
                        strides=8,
                        kernel_size=kernel_size)
    # 32x32 x 64+64 
    outputs3 = Conv2DTranspose(channels,
                               kernel_size=kernel_size,
                               strides=8,
                               padding='same')(d31)

    y = concatenate([outputs0, outputs1, outputs2, outputs3])
    y = Conv2DTranspose(channels,
                        kernel_size=1,
                        strides=1,
                        padding='same')(y)
    y = Conv2DTranspose(channels,
                        kernel_size=kernel_size,
                        strides=1,
                        activation='sigmoid',
                        padding='same')(y)
    outputs = y

    generator = Model(inputs, outputs, name=name)

    return generator


def build_discriminator(input_shape,
                        output_shape,
                        kernel_size=3,
                        name=None):

    inputs = Input(shape=input_shape)
    outputs = Input(shape=output_shape)
    x = concatenate([inputs, outputs])
    x = encoder_layer(x,
                      32,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    # 128x128x32
    x = encoder_layer(x,
                      64,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    # 64x64x64
    x = encoder_layer(x,
                      128,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    # 32x32x128
    x = encoder_layer(x,
                      256,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    # 16x16x256
    x = encoder_layer(x,
                      512,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    # 8x8x512
    x = encoder_layer(x,
                      1024,
                      kernel_size=kernel_size,
                      strides=1,
                      activation='leaky_relu',
                      instance_norm=False)
    # 8x8x1024

    x = Flatten()(x)
    x = Dense(1)(x)
    preal = Activation('linear')(x)
    discriminator = Model([inputs, outputs], preal, name=name)

    return discriminator


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



