"""Model builder

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Conv2DTranspose, Dropout
from keras.layers import LeakyReLU
from keras.models import Model
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
                  instance_norm=True,
                  postfix=None):

    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  name='conv_'+postfix)

    x = inputs
    if instance_norm:
        x = InstanceNormalization(name="in_"+postfix)(x)
    if activation == 'relu':
        x = Activation(activation, name='relu_'+postfix)(x)
    else:
        x = LeakyReLU(alpha=0.2, name='leaky_'+postfix)(x)
    x = conv(x)
    return x


def decoder_layer(inputs,
                  paired_inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  activation='relu',
                  instance_norm=True,
                  postfix=None):

    conv = Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding='same',
                           name='tconv_'+postfix)

    x = inputs
    if instance_norm:
        x = InstanceNormalization(name='in_'+postfix)(x)
    if activation == 'relu':
        x = Activation(activation, name='relu_'+postfix)(x)
    else:
        x = LeakyReLU(alpha=0.2, name='lrelu_'+postfix)(x)
    x = conv(x)
    x = concatenate([x, paired_inputs], name='concat_'+postfix)
    return x


def build_model(input_shape,
                output_shape=None,
                kernel_size=3,
                name='pspu_skelnet'):

    channels = int(output_shape[-1])

    inputs = Input(shape=input_shape)
    e11 = encoder_layer(inputs,
                        32,
                        strides=1,
                        kernel_size=kernel_size,
                        postfix='e11')
    # 256x256 x 32
    e12 = encoder_layer(e11,
                        64,
                        kernel_size=kernel_size,
                        postfix='e12')
    # 128x128 x 64
    e13 = encoder_layer(e12,
                        128,
                        kernel_size=kernel_size,
                        postfix='e13')
    # 64x64 x 128
    e14 = encoder_layer(e13,
                        256,
                        kernel_size=kernel_size,
                        postfix='e14')
    # 32x32 x 256
    e15 = encoder_layer(e14,
                        512,
                        kernel_size=kernel_size,
                        postfix='e15')
    # 16x16 x 512
    e16 = encoder_layer(e15,
                        1024,
                        kernel_size=kernel_size,
                        postfix='e16')
    # 8x8 x 1024
    e17 = encoder_layer(e16,
                        2048,
                        kernel_size=kernel_size,
                        postfix='e17')
    # 4x4 x 2048


    d11 = decoder_layer(e17,
                        e16,
                        1024,
                        kernel_size=kernel_size,
                        postfix='d11')
    # 8x8 x 1024+1024 
    d12 = decoder_layer(d11,
                        e15,
                        512,
                        kernel_size=kernel_size,
                        postfix='d12')
    # 16x16 x 512+512
    d13 = decoder_layer(d12,
                        e14,
                        256,
                        kernel_size=kernel_size,
                        postfix='d13')
    # 32x32 x 256+256
    d14 = decoder_layer(d13,
                        e13,
                        128,
                        kernel_size=kernel_size,
                        postfix='d14')
    # 64x64 x 128+128
    d15 = decoder_layer(d14,
                        e12,
                        64,
                        kernel_size=kernel_size,
                        postfix='d15')
    # 128x128 x 64+64
    d16 = decoder_layer(d15,
                        e11,
                        32,
                        kernel_size=kernel_size,
                        postfix='d16')
    # 256x256 x 32+32

    o1 = Conv2DTranspose(channels,
                         kernel_size=1,
                         strides=1,
                         padding='same',
                         name='tconv_o1')(d16)
    # 256x256 x channels


    e21 = encoder_layer(inputs,
                        32,
                        strides=1,
                        kernel_size=kernel_size,
                        postfix='e21')
    # 256x256 x 32
    e22 = encoder_layer(e21,
                        64,
                        strides=4,
                        kernel_size=kernel_size,
                        postfix='e22')
    # 64x64 x 64
    e23 = encoder_layer(e22,
                        128,
                        strides=4,
                        kernel_size=kernel_size,
                        postfix='e23')
    # 16x16 x 128
    e24 = encoder_layer(e23,
                        256,
                        strides=4,
                        kernel_size=kernel_size,
                        postfix='e24')
    # 4x4 x 256

    d21 = decoder_layer(e24,
                        e23,
                        128,
                        strides=4,
                        kernel_size=kernel_size,
                        postfix='d21')
    # 16x16 x 128+128
    d22 = decoder_layer(d21,
                        e22,
                        64,
                        strides=4,
                        kernel_size=kernel_size,
                        postfix='d22')
    # 64x64 x 64+64
    d23 = decoder_layer(d22,
                        e21,
                        32,
                        strides=4,
                        kernel_size=kernel_size,
                        postfix='d23')
    # 256x256 x 32+328
    o2 = Conv2DTranspose(channels,
                         kernel_size=1,
                         strides=1,
                         padding='same',
                         name='tconv_o2')(d23)
    # 256x256 x 1

    e31 = encoder_layer(inputs,
                        32,
                        strides=1,
                        kernel_size=kernel_size,
                        postfix='e31')
    # 256x256 x 32
    e32 = encoder_layer(e31,
                        64,
                        strides=8,
                        kernel_size=kernel_size,
                        postfix='e32')
    # 32x32 x 64
    e33 = encoder_layer(e32,
                        128,
                        strides=8,
                        kernel_size=kernel_size,
                        postfix='e33')
    # 4x4 x 128 

    d31 = decoder_layer(e33,
                        e32,
                        64,
                        strides=8,
                        kernel_size=kernel_size,
                        postfix='d31')
    # 32x32 x 64+64 
    d32 = decoder_layer(d31,
                        e31,
                        32,
                        strides=8,
                        kernel_size=kernel_size,
                        postfix='d32')
    # 256x256 x 32+32 
    o3 = Conv2DTranspose(channels,
                         kernel_size=1,
                         strides=1,
                         padding='same',
                         name='tconv_o3')(d32)

    y = concatenate([o1, o2, o3])
    y = Conv2DTranspose(32,
                        kernel_size=1,
                        strides=1,
                        padding='same',
                        name='tconv_pre')(y)
    y = Conv2DTranspose(channels,
                        kernel_size=kernel_size,
                        strides=1,
                        activation='sigmoid',
                        padding='same',
                        name='tconv_out')(y)
    outputs = y

    model = Model(inputs, outputs, name=name)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    input_shape = (256, 256, 1)
    output_shape = (256, 256, 1)
    model = build_model(input_shape, output_shape)
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)



