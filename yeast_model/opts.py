'''
Opts file for loading images - specifies the image sizes and paths to load data from and save weights to.

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

import os

batch_size = 64			# Batch size to use during training - if you run into memory constraints, reduce this
im_h = 64			# Height of input images
im_w = 64			# Width of input images

learning_rate = 1e-4
epochs = 30

checkpoint_path = './pretrained_weights/'	# Path to save the weights in after training
data_path = './toy_dataset/images/'		# Path to get image data

if checkpoint_path != '' and not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
