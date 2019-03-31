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
    x = Dropout(0.5)(x)
    x = concatenate([x, paired_inputs])
    return x


def build_generator(input_shape,
                    output_shape=None,
                    kernel_size=3,
                    name='pts_unet'):

    channels = int(output_shape[-1])

    inputs1 = Input(shape=input_shape)
    e11 = encoder_layer(inputs1,
                        128,
                        strides=1,
                        instance_norm=False,
                        kernel_size=kernel_size)
    # 256x256 x 64
    e12 = encoder_layer(e11,
                        256,
                        kernel_size=kernel_size)
    # 128x128 x 128
    e13 = encoder_layer(e12,
                        512,
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
                        1024,
                        kernel_size=kernel_size)
    # 4x4 x 512
    e18 = encoder_layer(e17,
                        1024,
                        kernel_size=kernel_size)
    # 2x2 x 512


    d11 = decoder_layer(e18,
                        e17,
                        1024,
                        kernel_size=kernel_size)
    # 4x4 x 512+512
    d12 = decoder_layer(d11,
                        e16,
                        512,
                        kernel_size=kernel_size)
    # 8x8 x 1024+512
    d13 = decoder_layer(d12,
                        e15,
                        512,
                        kernel_size=kernel_size)
    # 16x16 x 1024+512
    d14 = decoder_layer(d13,
                        e14,
                        512,
                        kernel_size=kernel_size)
    # 32x32 x 1024+512
    d15 = decoder_layer(d14,
                        e13,
                        512,
                        kernel_size=kernel_size)
    # 64x64 x 1024+256
    d16 = decoder_layer(d15,
                        e12,
                        256,
                        kernel_size=kernel_size)
    # 128x128 512+128
    d17 = decoder_layer(d16,
                        e11,
                        128,
                        kernel_size=kernel_size)
    # 256x256 256+64

    o1 = Conv2DTranspose(channels,
                         kernel_size=1,
                         strides=1,
                         padding='same')(d17)
    # 256x256 x channels


    inputs2 = Input(shape=input_shape)
    e21 = encoder_layer(inputs2,
                        128,
                        strides=1,
                        instance_norm=False,
                        kernel_size=kernel_size)
    # 256x256 x 64
    e22 = encoder_layer(e21,
                        256,
                        strides=4,
                        kernel_size=kernel_size)
    # 64x64 x 128
    e23 = encoder_layer(e22,
                        512,
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
                        256,
                        strides=4,
                        kernel_size=kernel_size)
    # 64x64 x 1024+128
    d23 = decoder_layer(d22,
                        e21,
                        128,
                        strides=4,
                        kernel_size=kernel_size)
    # 256x256 x 1024+64
    o2 = Conv2DTranspose(channels,
                         kernel_size=1,
                         strides=1,
                         padding='same')(d23)
    # 256x256 x 1

    inputs3 = Input(shape=input_shape)
    e31 = encoder_layer(inputs3,
                        128,
                        strides=1,
                        instance_norm=False,
                        kernel_size=kernel_size)
    # 256x256 x 64
    e32 = encoder_layer(e31,
                        256,
                        strides=8,
                        kernel_size=kernel_size)
    # 32x32 x 128
    e33 = encoder_layer(e32,
                        512,
                        strides=8,
                        kernel_size=kernel_size)
    # 4x4 x 256

    d31 = decoder_layer(e33,
                        e32,
                        256,
                        strides=8,
                        kernel_size=kernel_size)
    # 32x32 x 128+128 
    d32 = decoder_layer(d31,
                        e31,
                        128,
                        strides=8,
                        kernel_size=kernel_size)
    # 256x256 x 64+64
    o3 = Conv2DTranspose(channels,
                         kernel_size=1,
                         strides=1,
                         padding='same')(d32)

    y = concatenate([o1, o2, o3])
    y = Conv2DTranspose(channels,
                        kernel_size=kernel_size,
                        strides=1,
                        activation='sigmoid',
                        padding='same')(y)
    outputs = y

    generator = Model([inputs1, inputs2, inputs3], outputs, name=name)
    generator.summary()

    return generator
