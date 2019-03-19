"""Model trainer

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import argparse
import os
from model import build_generator, build_discriminator
from skimage.io import imsave
from utils import list_files, read_gray, augment
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam, RMSprop
from keras.models import Model
from keras.layers import Input
import datetime

from other_utils import test_generator, display_images


PT_PATH = "dataset/pixel/train"
PX_PATH = "dataset/pixel/test"
PR_PATH = "dataset/pixel/root"
EPOCHS = 200

def predict_pix(model, path=PX_PATH, ispt=False):
    if ispt:
        path = path.replace("pixel", "point")
        path = path.replace("test", "test_img")
    files = list_files(path)
    pix = []
    for f in files:
        pix_file = os.path.join(path, f)
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
        out_pix = generator.predict([pix, pix, pix, pix])
        print("Max: ", np.amax(pix))
        out_pix[out_pix>=0.2] = 1.0
        out_pix[out_pix<0.1] = 0.0
        out_pix = np.squeeze(out_pix) * 255.0
        out_pix = out_pix.astype(np.uint8)
        print(out_pix.shape)
        path = os.path.join(PR_PATH, files[i])
        if ispt:
            path = path.replace("pixel", "point")
        print("Saving ... ", path)
        if ispt:
            imsave(path, out_pix, cmap='gray')
        else:
            out_pix = np.expand_dims(out_pix, axis=2)
            out_pix = np.concatenate((out_pix, out_pix, out_pix), axis=2)
            imsave(path, out_pix)


def train(models, source_data, target_data, batch_size=8):

    # the models
    generator, discriminator, adv = models
    # network parameters
    # batch_size, train_steps, patch, model_name = params
    # train dataset

    # the generator image is saved every 2000 steps
    save_interval = 500
    target_size = target_data.shape[0]
    source_size = source_data.shape[0]

    valid = np.ones([batch_size, 1])
    fake = np.zeros([batch_size, 1])

    valid_fake = np.concatenate((valid, fake))
    valid_valid = np.concatenate((valid, valid))
    start_time = datetime.datetime.now()
    train_steps = 400000

    rand_indexes = np.random.randint(0, source_size, size=16)
    test_data = source_data[rand_indexes]
    display_images(test_data, filename="source.png")


    for step in range(train_steps):
        # sample a batch of real target data
        rand_indexes = np.random.randint(0, target_size, size=batch_size)
        real_target = target_data[rand_indexes]
        real_source = source_data[rand_indexes]

        # sample a batch of real source data
        rand_indexes = np.random.randint(0, source_size, size=batch_size)
        fake_source = source_data[rand_indexes]
        # generate a batch of fake target data fr real source data
        fake_target = generator.predict(fake_source)
        # fake_pair = [fake_source, fake_target]
        
        real_fake_source = np.concatenate( (real_source, fake_source) )
        # print(real_fake_source.shape)
        # print(valid_fake.shape)
        real_fake_target = np.concatenate( (real_target, fake_target) )
        # combine real and fake into one batch
        # x = np.concatenate((real_pair, fake_pair))
        # train the target discriminator using fake/real data
        metrics = discriminator.train_on_batch([real_fake_source, real_fake_target], valid_fake)
        log = "%d: [d_target loss: %f]" % (step, metrics)

        rand_indexes = np.random.randint(0, source_size, size=2*batch_size)
        real_source = source_data[rand_indexes]
        real_target = target_data[rand_indexes]
        metrics = adv.train_on_batch(real_source, [valid_valid, real_target])
        #fmt = "%s [adv loss: %f] "
        #log = fmt % (log, metrics[0])

        #rand_indexes = np.random.randint(0, target_size, size=batch_size)
        #real_target = target_data[rand_indexes]
        #real_source = source_data[rand_indexes]
        #metrics = generator.train_on_batch(real_source, real_target)

        elapsed_time = datetime.datetime.now() - start_time
        fmt = "%s [net loss: %f] [adv loss: %f] [gen loss: %f] [time: %s]"
        log = fmt % (log, metrics[0], metrics[1], metrics[2], elapsed_time)
        print(log)
        if (step + 1) % save_interval == 0 or step == 0:
            test_generator(generator,
                           test_data,
                           step=step+1)
            # save the models after training the generators
            generator.save_weights("generator.h5")
            discriminator.save_weights("discriminator.h5")


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 160:
        lr = 0.5e-4
    elif epoch > 120:
        lr = 1e-4
    elif epoch > 80:
        lr = 0.5e-3
    print('Learning rate: ', lr)
    return lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Gen weights"
    parser.add_argument("--gen",
                        default=None,
                        help=help_)
    help_ = "Dis weights"
    parser.add_argument("--dis",
                        default=None,
                        help=help_)
    help_ = "Train"
    parser.add_argument("--train",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Plot model"
    parser.add_argument("--plot",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Aug"
    parser.add_argument("--aug",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Batch size"
    parser.add_argument("--batch_size", type=int, default=8, help=help_)

    help_ = "Number of GPUs (default is 1)"
    parser.add_argument("--gpus", type=int, default=1, help=help_)

    args = parser.parse_args()

    infile = "in_pts.npy"
    outfile = "out_pts.npy"
    print("Loading ... ", infile) 
    input_pix = np.load(infile)
    print("Loading ... ", outfile) 
    output_pix = np.load(outfile)

    print("batch size: ", args.batch_size)
    input_shape = input_pix.shape[1:]
    output_shape = output_pix.shape[1:]

    generator = build_generator(input_shape, output_shape, kernel_size=3)
    generator.summary()

    if args.plot:
        from keras.utils import plot_model
        plot_model(generator, to_file='generator.png', show_shapes=True)

    if args.gen is not None:
        print("Loading generator weights ...", args.gen)
        generator.load_weights(args.gen)

    if not args.train:
        predict_pix(generator, ispt=True)
    else:
        #optimizer = RMSprop(lr=2e-4)
        #discriminator.compile(loss='mse', optimizer=optimizer)

        #discriminator.trainable = False
        #source_input = Input(shape=input_shape)
        #outputs = generator(source_input)
        #adversarial = Model(source_input, [discriminator([source_input, generator(source_input)]), outputs], name="adv")
        #optimizer = RMSprop(lr=1e-4)
        #loss_weights = [1.0, 2.0]
        #loss = ['mse', 'binary_crossentropy']
        # adversarial.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        #adversarial.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)
        #adversarial.summary()

        # train discriminator and adversarial networks
        # models = (generator, discriminator, adversarial)
        #train(models, input_pix, output_pix, args.batch_size)

        optimizer = Adam(lr=1e-3)
        generator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'weights')
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

        # train the model with input images and labels
        for i in range(EPOCHS):
            x, y = augment(input_pix, output_pix)
            x = np.concatenate((input_pix, x), axis=0)
            y = np.concatenate((output_pix, y), axis=0)
            print("Augmented input shape: ", x.shape)
            print("Augmented output shape: ", y.shape)
            x = x.astype('float32') / 255
            y = y.astype('float32') / 255
            inputs = [x, x, x, x]
            generator.fit(inputs,
                          y,
                          epochs=8,
                          batch_size=args.batch_size,
                          callbacks=callbacks)

        optimizer = Adam(lr=1e-4)
        generator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
        for i in range(EPOCHS):
            x, y = augment(input_pix, output_pix)
            x = np.concatenate((input_pix, x), axis=0)
            y = np.concatenate((output_pix, y), axis=0)
            x = x.astype('float32') / 255
            y = y.astype('float32') / 255
            inputs = [x, x, x, x]
            generator.fit(inputs,
                          y,
                          epochs=1,
                          batch_size=args.batch_size,
                          callbacks=callbacks)

