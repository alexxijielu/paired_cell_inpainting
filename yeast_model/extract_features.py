'''
Given a directory of subfolders of single cell image crops, extract features for every single cell in that directory.

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from PIL import Image
import tensorflow as tf
import keras
import skimage.exposure

import opts as opt
from dataset import Dataset
from pair_model import Pair_Model

if __name__ == "__main__":
    # Layers to extract single cell features from
    layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

    # Directory of subfolders of single-cell image crops
    datapath = opt.data_path

    # Location of pretrained weights for the model
    modelpath = opt.checkpoint_path + "model_weights.h5"

    for layer in layers:
        print ("Loading the model...")
        # Load pretrained model and set the layer to extract features from
        model = Pair_Model().create_model((opt.im_h, opt.im_w, 2), (opt.im_h, opt.im_w, 1))
        model.load_weights(modelpath)
        intermediate_model = tf.keras.Model(inputs=model.get_layer("x_in").input,
                                            outputs=model.get_layer(layer).output)

        print ("Evaluating images...")
        # Load each single cell and extract features into a file
        for dir in os.listdir(datapath):
            for image in os.listdir(datapath + dir):
                if "_gfp.tif" in image:
                    print ("Evaluating", image)
                    # Iterate over every single cell crop and preprocess it
                    name = image.rsplit("_", 1)[0]
                    gfp = np.array(Image.open(datapath + dir + "/" + image)).astype(np.float32)
                    rfp = np.array(Image.open(datapath + dir + "/" + image.replace("gfp", "rfp"))).astype(np.float32)

                    gfp = skimage.exposure.rescale_intensity(gfp, out_range=(0, 1))
                    rfp = skimage.exposure.rescale_intensity(rfp, out_range=(0, 1))

                    # Feed single cell crop into the pretrained model and obtain features
                    x_in = np.stack((gfp, rfp), axis=-1)
                    x_in = np.expand_dims(x_in, axis=0)

                    prediction = intermediate_model.predict([x_in], batch_size=1)

                    prediction = np.squeeze(prediction)
                    prediction = np.max(prediction, axis=(0, 1))

                    # Write features into a file
                    outputfile = opt.checkpoint_path + "yeast_features_" + layer + ".txt"
                    output = open(outputfile, "a")
                    output.write(name)
                    output.write("\t")
                    for feat in prediction:
                        output.write(str(feat))
                        output.write("\t")
                    output.write("\n")
                    output.close()
