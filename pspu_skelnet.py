"""PSPU-SkelNet: build, train, test

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import argparse
import os
import datetime
from skimage.io import imsave

from model_builder import build_model
from utils import list_files, read_gray, augment

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras.layers import Input


TEST_PATH = "dataset/point/test_img"
PRED_PATH = "dataset/point/root"
EPOCHS = 140


class PSPU_SkelNet():
    def __init__(self,
                 batch_size=8,
                 ntimes=8):
        self.thresh = 0.5
        self.batch_size = batch_size
        self.ntimes = ntimes
        self.load_train_data()
        self.build_model()


    def load_train_data(self):
        infile = "npy/in_pts.npy"
        outfile = "npy/out_pts.npy"
        print("Loading input train data... ", infile) 
        self.input_pix = np.load(infile)
        print("Input train data shape: ", self.input_pix.shape)
        print("Loading output train data ... ", outfile) 
        self.output_pix = np.load(outfile)
        print("Output train data shape: ", self.output_pix.shape)


    def build_model(self):
        input_shape = self.input_pix.shape[1:]
        output_shape = self.output_pix.shape[1:]
        self.model = build_model(input_shape, output_shape)
        self.model.summary()
        optimizer = Adam(lr=1e-3)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])


    def plot_model(self):
        from keras.utils import plot_model
        plot_model(self.model, to_file='pspu_skelnet.png', show_shapes=True)


    def load_weights(self, weights_file):
        print("Loading model weights ...", weights_file)
        self.model.load_weights(weights_file)


    def lr_schedule(self, epoch):
        lr = 1e-3
        if epoch > 60:
            lr = 0.5e-5
        elif epoch > 20:
            lr = 1e-4
        print('Learning rate: ', lr)
        return lr


    def train(self):
        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'weights')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        weights_name = 'pspu_skelnet.h5' 
        filepath = os.path.join(save_dir, weights_name)

        # prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     verbose=1,
                                     save_weights_only=True)
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        callbacks = [checkpoint, lr_scheduler]

        # train the model with input images and labels
        xval = self.input_pix.astype('float32') / 255
        yval = self.output_pix.astype('float32') / 255

        x, y = augment(self.input_pix, self.output_pix, ntimes=self.ntimes)
        x = np.concatenate((self.input_pix, x), axis=0)
        y = np.concatenate((self.output_pix, y), axis=0)
        print("Augmented input train data shape: ", x.shape)
        print("Augmented output train data shape: ", y.shape)
        x = x.astype('float32') / 255
        y = y.astype('float32') / 255
        self.model.fit(x,
                       y,
                       epochs=EPOCHS,
                       validation_data=(xval, yval),
                       batch_size=self.batch_size,
                       callbacks=callbacks)


    def predict(self):
        path = TEST_PATH
        files = list_files(path)
        pix = []
        for f in files:
            pix_file = os.path.join(path, f)
            pix_data =  read_gray(pix_file)
            pix.append(pix_data)
            print(pix_file)

        pix = np.array(pix)
        pix = np.expand_dims(pix, axis=3)
        pix = pix / 255.0

        for i in range(pix.shape[0]):
            img = pix[i]
            img = np.expand_dims(img, axis=0)
            out_pix = self.model.predict(img)
            out_pix[out_pix >= self.thresh] = 1.0
            out_pix[out_pix < self.thresh] = 0.0
            out_pix = np.squeeze(out_pix) * 255.0
            out_pix = out_pix.astype(np.uint8)
            path = os.path.join(PRED_PATH, files[i])
            print("Saving ... ", path)
            imsave(path, out_pix, cmap='gray')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load model saved weights"
    parser.add_argument("--weights",
                        default=None,
                        help=help_)
    help_ = "Train model"
    parser.add_argument("--train",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Plot model"
    parser.add_argument("--plot",
                        default=False,
                        action='store_true',
                        help=help_)

    help_ = "Number of times (rotate, translate, etc) is executed"
    parser.add_argument("--ntimes", type=int, default=8, help=help_)

    help_ = "Batch size"
    parser.add_argument("--batch_size", type=int, default=8, help=help_)

    args = parser.parse_args()

    pspu_skelnet = PSPU_SkelNet(batch_size=args.batch_size,
                                ntimes=args.ntimes)
    print("Batch size: ", args.batch_size)

    if args.plot:
        pspu_skelnet.plot_model()

    if args.weights is not None:
        pspu_skelnet.load_weights(args.weights)

    if not args.train:
        pspu_skelnet.predict()
    else:
        pspu_skelnet.train()

