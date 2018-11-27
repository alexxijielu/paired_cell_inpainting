'''
Opts file for loading images - specifies the image sizes and paths to load data from and save weights to.

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

import os

batch_size = 4		# Batch size to use during training - if you run into memory constraints, reduce this
                    # For batch size, if you're using the toy dataset, set this to a small number to avoid errors
im_h = 64		# Height of input images
im_w = 64		# Width of input images

checkpoint_path = './pretrained_weights/'		    # Path to save the weights in after training
data_path = './toy_dataset/'	                    # Path to get image data from (single cell crops)

learning_rate = 1e-4
epochs = 30

if checkpoint_path != '' and not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
