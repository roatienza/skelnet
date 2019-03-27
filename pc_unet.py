'''Point cloud stacked autoencoder.

'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.layers import Dense, Input, Activation
from keras.layers import Conv1D, Flatten 
from keras.layers import Reshape, UpSampling1D, BatchNormalization, MaxPooling1D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import RMSprop
from keras.layers.merge import concatenate
from keras.layers.merge import concatenate

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import datetime
import sys

sys.path.append("external")


def encoder_layer(x, filters, strides=1, dilation_rate=1):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=filters,
               kernel_size=3,
               strides=strides,
               dilation_rate=dilation_rate,
               padding='same')(x)
    return x

def decoder_layer(x, pair, filters, dilation_rate=1):
    x = encoder_layer(x, filters, dilation_rate=dilation_rate)
    x = UpSampling1D()(x)
    x = concatenate([x, pair])
    return x

def compression_layer(x, y, maxpool=True):
    if maxpool:
        y = MaxPooling1D()(y)
    x = concatenate([x, y])

    y = Conv1D(filters=64,
               kernel_size=1,
               activation='relu',
               padding='same')(x)
    return x, y
        
def build_model(input_shape, filters=64, activation='linear'):

    inputs = Input(shape=input_shape)
    inputs = Input(shape=input_shape, name='encoder_input')
    e1 = encoder_layer(inputs, 32, strides=1)
    e2 = encoder_layer(e1, 64, strides=2)
    e3 = encoder_layer(e2, 128, strides=2)
    e4 = encoder_layer(e3, 256, strides=2)
    e5 = encoder_layer(e4, 512, strides=2)

    d0 = decoder_layer(e5, e4, 256)
    d1 = decoder_layer(d0, e3, 128)
    d2 = decoder_layer(d1, e2, 64)
    d3 = decoder_layer(d2, e1, 32)
    outputs = Conv1D(filters=32,
                     kernel_size=3,
                     padding='same')(d3)
    outputs = Conv1D(filters=1,
                     kernel_size=1,
                     padding='same')(outputs)
    model = Model(inputs, outputs)
    model.summary()
    return model
    

def loss(gt, pred):
    from structural_losses import tf_nndistance
    p1top2 , _, p2top1, _ = tf_nndistance.nn_distance(pred, gt)
    cd_loss = K.mean(p1top2) + K.mean(p2top1)
    return cd_loss


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 100:
        lr = 1e-5
    elif epoch > 60:
        lr = 1e-4
    print('Learning rate: ', lr)
    return lr



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use Chamder distance loss"
    parser.add_argument("--chamfer", default=False, action='store_true', help=help_)
    help_ = "Shapnet category or class (chair, airplane, etc)"
    parser.add_argument("-a", "--category", default='all', help=help_)
    args = parser.parse_args()

    maxpts = 1024 * 12
    input_shape = (maxpts, 3)
    model = build_model(input_shape)
    optimizer = RMSprop(lr=1e-3)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'weights')
    model_name = 'skelnet_pc_model.h5' 
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 verbose=1,
                                 save_weights_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    callbacks = [checkpoint, lr_scheduler]

    exit(0)
    print("Augmented input shape: ", x.shape)
    print("Augmented output shape: ", y.shape)
    x = x.astype('float32') / 255
    y = y.astype('float32') / 255
    model.fit(x, y, epochs=100, batch_size=32, callbacks=callbacks)