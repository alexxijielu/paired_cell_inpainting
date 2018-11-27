'''
Script to train a human paired cell inpainting model from scratch.

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
from skimage.transform import resize

import opts as opt
from dataset import Dataset
from pair_model import Pair_Model


'''Given a dataset class (see dataset.py), load an image from the class'''
def load_image_gt(ds, image_id, augment=True):
    # Load image and pair
    x_antibody, x_nucleus, x_microtubule = ds.load_image(image_id)
    y_antibody, y_nucleus, y_microtubule = ds.sample_pair_equally(image_id)

    x_antibody = resize(x_antibody, (64, 64))
    x_nucleus = resize(x_nucleus, (64, 64))
    x_microtubule = resize(x_microtubule, (64, 64))
    y_antibody = resize(y_antibody, (64, 64))
    y_nucleus = resize(y_nucleus, (64, 64))
    y_microtubule = resize(y_microtubule, (64, 64))

    # Whatever preprocessing operations we need here
    # Rescale images
    x_antibody = skimage.exposure.rescale_intensity(x_antibody.astype(np.float32), out_range=(0, 1))
    y_antibody = skimage.exposure.rescale_intensity(y_antibody.astype(np.float32), out_range=(0, 1))
    x_nucleus = skimage.exposure.rescale_intensity(x_nucleus.astype(np.float32), out_range=(0, 1), in_range=(0.05, 1.0))
    y_nucleus = skimage.exposure.rescale_intensity(y_nucleus.astype(np.float32), out_range=(0, 1), in_range=(0.05, 1.0))
    x_microtubule = skimage.exposure.rescale_intensity(x_microtubule.astype(np.float32), out_range=(0, 1), in_range=(0.05, 1.0))
    y_microtubule = skimage.exposure.rescale_intensity(y_microtubule.astype(np.float32), out_range=(0, 1), in_range=(0.05, 1.0))

    # Randomly flip the images if augmenting
    if augment:
        if np.random.choice([0, 1]):
            x_antibody = np.fliplr(x_antibody)
            x_nucleus = np.fliplr(x_nucleus)
            x_microtubule = np.fliplr(x_microtubule)
        if np.random.choice([0, 1]):
            x_antibody = np.flipud(x_antibody)
            x_nucleus = np.flipud(x_nucleus)
            x_microtubule = np.flipud(x_microtubule)
        if np.random.choice([0, 1]):
            y_antibody = np.fliplr(y_antibody)
            y_nucleus = np.fliplr(y_nucleus)
            y_microtubule = np.fliplr(y_microtubule)
        if np.random.choice([0, 1]):
            y_antibody = np.flipud(y_antibody)
            y_nucleus = np.flipud(y_nucleus)
            y_microtubule = np.flipud(y_microtubule)

    # Stack inputs and outputs as necessary
    x_in = np.stack((x_antibody, x_nucleus, x_microtubule), axis=-1)
    y_in = np.stack((y_nucleus, y_microtubule), axis=-1)
    y_out = np.expand_dims(y_antibody, axis=-1)

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
    model = Pair_Model().create_model((opt.im_h, opt.im_w, 3), (opt.im_h, opt.im_w, 2))

    optimizer = tf.train.AdamOptimizer(learning_rate=opt.learning_rate, beta1=0.5)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit_generator(train_generator, steps_per_epoch=steps, epochs=opt.epochs, workers=40, max_queue_size=150,
                        use_multiprocessing=True)

    print("Saving model weights in " + opt.checkpoint_path)
    # Save the model weights
    model.save(opt.checkpoint_path + "model_weights.h5")