'''
A script to extract features using a pretrained VGG16 model on Imagenet - used for benchmarking purposes.

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

import tensorflow as tf
import os, sys
import numpy as np
from PIL import Image

from skimage.transform import resize

# Layers to extract features from VGG16 from; we systematically tested each layer and reported highest performance
layers = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1',
          'block3_conv2', 'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3']

# Directory of single-cell image crop folders to extract features from
datapath = "../human_protein_atlas_single_cell/"

# Output directory to write features to
output_fold = "../vgg16_features/"

# Create the output directory if necessary
if not os.path.exists(output_fold):
    os.makedir(output_fold)

# Construct the VGG16 model for each layer and extract all single cell features from that layer into a separate
# file in the output directory
for layer in layers:
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output)

    for directory in os.listdir(datapath):
        for image in os.listdir(datapath + directory):
            if ("_green" in image):
                # We tested several strategies for transfer learning:
                # The best performing is to input the protein channel only as a greyscale image

                # Get each image and preprocess the image
                print("Predicting " + image)
                x_batch = np.array(Image.open(datapath + directory + "/" + image))

                x_batch = resize(x_batch, output_shape=(64, 64))
                x_batch = np.rint(x_batch * 255)

                x_batch = np.expand_dims(x_batch, -1)
                x_batch = np.tile(x_batch, 3)

                x_batch = np.expand_dims(x_batch, 0)
                x_batch = preprocess_input(x_batch, data_format='channels_last')

                # Extract feature map and write features to file
                pred = model.predict(x_batch)
                pred = np.squeeze(pred)
                pred = np.max(pred, axis=(0, 1))

                id = image.split(".tif")[0]
                outputfile = output_fold + layer + "_green.txt"                
                output = open(outputfile, "a")
                output.write(directory)
                output.write("\t")
                for feat in pred:
                    output.write(str(feat))
                    output.write("\t")
                output.write("\n")
                output.close()
