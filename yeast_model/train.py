'''
Script to train a yeast paired cell inpainting model from scratch.

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tensorflow as tf
import keras
import skimage.exposure

import opts as opt
from dataset import Dataset
from pair_model import Pair_Model

'''Given a dataset class (see dataset.py), load an image from the class'''
def load_image_gt(ds, image_id, augment=True):
    # Load image and pair
    x_protein, x_bf, label = ds.load_image_with_label(image_id)
    y_protein, y_bf = ds.sample_pair_equally(image_id)

    # Whatever preprocessing operations we need here
    # Rescale images
    with warnings.catch_warnings():
        try:
            x_protein = skimage.exposure.rescale_intensity(x_protein.astype(np.float32), out_range=(0, 1))
            y_protein = skimage.exposure.rescale_intensity(y_protein.astype(np.float32), out_range=(0, 1))
            x_bf = skimage.exposure.rescale_intensity(x_bf.astype(np.float32), out_range=(0, 1))
            y_bf = skimage.exposure.rescale_intensity(y_bf.astype(np.float32), out_range=(0, 1))
        except RuntimeWarning:
            print (label)

    # Randomly flip the images if augmenting
    if augment:
        if np.random.choice([0, 1]):
            x_protein = np.fliplr(x_protein)
            x_bf = np.fliplr(x_bf)
        if np.random.choice([0, 1]):
            x_protein = np.flipud(x_protein)
            x_bf = np.flipud(x_bf)
        if np.random.choice([0, 1]):
            y_protein = np.fliplr(y_protein)
            y_bf = np.fliplr(y_bf)
        if np.random.choice([0, 1]):
            y_protein = np.flipud(y_protein)
            y_bf = np.flipud(y_bf)

    # Stack inputs and outputs as necessary
    x_in = np.stack((x_protein, x_bf), axis=-1)
    y_in = np.expand_dims(y_bf, axis=-1)
    y_out = np.expand_dims(y_protein, axis=-1)

    return x_in, y_in, y_out

'''Data generator for Keras model (retrieves images infinitely)'''
def data_generator(dataset, shuffle=True, augment=True, batch_size=1):
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)

    # Runs indefinitely for Keras
    while True:
        # If we've exhausted the image dataset, reshuffle the indices
        image_index = (image_index + 1) % len(image_ids)
        if shuffle and image_index == 0:
            np.random.shuffle(image_ids)

        # Get current image
        image_id = image_ids[image_index]
        x_in, y_in, y_out = load_image_gt(dataset, image_id, augment=augment)

        # Initialize batch arrays if empty
        if b == 0:
            batch_x_in = np.zeros((batch_size,) + x_in.shape)
            batch_y_in = np.zeros((batch_size,) + y_in.shape)
            batch_y_out = np.zeros((batch_size,) + y_out.shape)

        # Add image to batch
        batch_x_in[b] = x_in
        batch_y_in[b] = y_in
        batch_y_out[b] = y_out
        b += 1

        # Check if the batch is full
        if b >= batch_size:
            inputs = [batch_x_in, batch_y_in]
            outputs = batch_y_out

            yield inputs, outputs

            # start a new batch
            b = 0


if __name__ == "__main__":
    print("Preparing the dataset...")
    # Load all images in the training set (argument given in opts.py) into a Dataset class and
    # create data generator for training
    ds = Dataset()
    ds.add_dataset(opt.data_path)
    ds.prepare()
    train_generator = data_generator(ds, batch_size=opt.batch_size)
    steps = len(ds.image_info) // opt.batch_size

    print("Training the model...")
    # Train the model (specify learning rates and epochs here)
    model = Pair_Model().create_model((opt.im_h, opt.im_w, 2), (opt.im_h, opt.im_w, 1))

    optimizer = tf.train.AdamOptimizer(learning_rate=opt.learning_rate, beta1=0.5)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit_generator(train_generator, steps_per_epoch=steps, epochs=opt.epochs, workers=40, max_queue_size=150,
                        use_multiprocessing=True)

    print("Saving model weights in " + opt.checkpoint_path)
    # Save the model weights
    model.save(opt.checkpoint_path + "model_weights.h5")

