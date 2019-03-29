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
                  kernel_size=4,
                  strides=2,
                  activation='lrelu',
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
                  kernel_size=4,
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
    x = Dropout(0.5)(x)
    x = concatenate([x, paired_inputs])
    return x


def build_generator(input_shape,
                    output_shape=None,
                    kernel_size=4,
                    name=None):

    channels = int(output_shape[-1])

    inputs1 = Input(shape=input_shape)
    e11 = encoder_layer(inputs1,
                        64,
                        strides=1,
                        instance_norm=False,
                        kernel_size=kernel_size)
    # 256x256 x 64
    e12 = encoder_layer(e11,
                        128,
                        kernel_size=kernel_size)
    # 128x128 x 128
    e13 = encoder_layer(e12,
                        256,
                        kernel_size=kernel_size)
    # 64x64 x 256
    e14 = encoder_layer(e13,
                        512,
                        kernel_size=kernel_size)
    # 32x32 x 512
    e15 = encoder_layer(e14,
                        512,
                        kernel_size=kernel_size)
    # 16x16 x 512
    e16 = encoder_layer(e15,
                        512,
                        kernel_size=kernel_size)
    # 8x8 x 512
    e17 = encoder_layer(e16,
                        512,
                        kernel_size=kernel_size)
    # 4x4 x 512
    e18 = encoder_layer(e17,
                        512,
                        kernel_size=kernel_size)
    # 2x2 x 512


    d11 = decoder_layer(e18,
                        e17,
                        512,
                        kernel_size=kernel_size)
    # 4x4 x 512+512
    d12 = decoder_layer(d11,
                        e16,
                        1024,
                        kernel_size=kernel_size)
    # 8x8 x 1024+512
    d13 = decoder_layer(d12,
                        e15,
                        1024,
                        kernel_size=kernel_size)
    # 16x16 x 1024+512
    d14 = decoder_layer(d13,
                        e14,
                        1024,
                        kernel_size=kernel_size)
    # 32x32 x 1024+512
    d15 = decoder_layer(d14,
                        e13,
                        1024,
                        kernel_size=kernel_size)
    # 64x64 x 1024+256
    d16 = decoder_layer(d15,
                        e12,
                        512,
                        kernel_size=kernel_size)
    # 128x128 512+128
    d17 = decoder_layer(d16,
                        e11,
                        256,
                        kernel_size=kernel_size)
    # 256x256 256+64

    o1 = Conv2DTranspose(channels,
                         kernel_size=kernel_size,
                         strides=1,
                         padding='same')(d17)
    # 256x256 x channels


    inputs2 = Input(shape=input_shape)
    e21 = encoder_layer(inputs2,
                        64,
                        strides=1,
                        instance_norm=False,
                        kernel_size=kernel_size)
    # 256x256 x 64
    e22 = encoder_layer(e21,
                        128,
                        strides=4,
                        kernel_size=kernel_size)
    # 64x64 x 128
    e23 = encoder_layer(e22,
                        256,
                        strides=4,
                        kernel_size=kernel_size)
    # 16x16 x 256
    e24 = encoder_layer(e23,
                        512,
                        strides=4,
                        kernel_size=kernel_size)
    # 4x4 x 512

    d21 = decoder_layer(e24,
                        e23,
                        512,
                        strides=4,
                        kernel_size=kernel_size)
    # 16x16 x 512+256
    d22 = decoder_layer(d21,
                        e22,
                        1024,
                        strides=4,
                        kernel_size=kernel_size)
    # 64x64 x 1024+128
    d23 = decoder_layer(d22,
                        e21,
                        1024,
                        strides=4,
                        kernel_size=kernel_size)
    # 256x256 x 1024+64
    o2 = Conv2DTranspose(channels,
                         kernel_size=kernel_size,
                         strides=1,
                         padding='same')(d23)
    # 256x256 x 1

    inputs3 = Input(shape=input_shape)
    e31 = encoder_layer(inputs3,
                        64,
                        strides=1,
                        instance_norm=False,
                        kernel_size=kernel_size)
    # 256x256 x 64
    e32 = encoder_layer(e31,
                        128,
                        strides=8,
                        kernel_size=kernel_size)
    # 32x32 x 128
    e33 = encoder_layer(e32,
                        256,
                        strides=8,
                        kernel_size=kernel_size)
    # 4x4 x 256

    d31 = decoder_layer(e33,
                        e32,
                        512,
                        strides=8,
                        kernel_size=kernel_size)
    # 32x32 x 512+128 
    d32 = decoder_layer(d31,
                        e31,
                        1024,
                        strides=8,
                        kernel_size=kernel_size)
    # 256x256 x 1024+64
    o3 = Conv2DTranspose(channels,
                         kernel_size=kernel_size,
                         strides=1,
                         padding='same')(d32)

    y = concatenate([o1, o2, o3])
    #y = Conv2DTranspose(32,
    #                    kernel_size=1,
    #                    strides=1,
    #                    padding='same')(y)
    y = Conv2DTranspose(channels,
                        kernel_size=5,
                        strides=1,
                        activation='sigmoid',
                        padding='same')(y)
    outputs = y

    generator = Model([inputs1, inputs2, inputs3], outputs, name=name)
    generator.summary()

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



