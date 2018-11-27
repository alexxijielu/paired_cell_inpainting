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
import skimage.exposure
from skimage.transform import resize

import opts as opt
from pair_model import Pair_Model

if __name__ == "__main__":
    datapath = "./toy_dataset/"         # Directory of subfolders of single-cell image crops
    target_layer = "conv3_1"            # Which layer of the trained model to extract features from
    model_weights = opt.checkpoint_path + "model_weights.h5"    # Location of pretrained weights for the model

    # Load pretrained model and set the layer to extract features from
    print ("Loading the model...")
    model = Pair_Model().create_model((opt.im_h, opt.im_w, 3), (opt.im_h, opt.im_w, 2))
    model.load_weights(model_weights)
    intermediate_model = tf.keras.Model(inputs=model.get_layer("x_in").input,
                                        outputs=model.get_layer(target_layer).output)

    # Load each single cell and extract features into a file
    print ("Evaluating images...")
    for indir in os.listdir(datapath):
        for image in os.listdir(datapath + indir):
            if "_green.tif" in image:
                print ("Evaluating", image)
                # Iterate over every single cell crop and preprocess it
                name = image.split("_green")[0]
                antibodyname = datapath + indir + "/" + image
                nucleusname = antibodyname.replace("green", "blue")
                microtubulename = antibodyname.replace("green", "red")

                antibody = np.array(Image.open(antibodyname))
                nucleus = np.array(Image.open(nucleusname))
                microtubule = np.array(Image.open(microtubulename))

                antibody = resize(antibody, (64, 64))
                nucleus = resize(nucleus, (64, 64))
                microtubule = resize(microtubule, (64, 64))

                antibody = skimage.exposure.rescale_intensity(antibody.astype(np.float32), out_range=(0, 1))
                nucleus = skimage.exposure.rescale_intensity(nucleus.astype(np.float32), out_range=(0, 1),
                                                             in_range=(0.05, 1.0))
                microtubule = skimage.exposure.rescale_intensity(microtubule.astype(np.float32),
                                                                 out_range=(0, 1), in_range=(0.05, 1.0))

                # Feed single cell crop into the pretrained model and obtain features
                x_in = np.stack((antibody, nucleus, microtubule), axis=-1)
                x_in = np.expand_dims(x_in, axis=0)

                prediction = intermediate_model.predict([x_in], batch_size=1)

                prediction = np.squeeze(prediction)
                prediction = np.max(prediction, axis=(0, 1))

                # Write features into a file
                if not os.path.exists(opt.checkpoint_path + "features/"):
                    os.mkdir(opt.checkpoint_path + "features/")

                outputfile = opt.checkpoint_path + "features/" + indir + ".txt"
                output = open(outputfile, "a")
                output.write(name)
                output.write("\t")
                for feat in prediction:
                    output.write(str(feat))
                    output.write("\t")
                output.write("\n")
                output.close()
