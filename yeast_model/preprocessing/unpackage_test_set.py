'''
Given the hdf5 files of annotated single yeast cell crops from Chong et al. 2015 (see README for instructions),
sort these into a directory appropriate for paired cell inpainting model input.

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

import deepdish as dd
import numpy as np
from PIL import Image
import os

# Path of any of the three hdf5 files just to get meta information
path = "../DeepLoc_full_datasets/datasets/Chong_test_set.hdf5"

# Path of all three hdf5 files - used to assemble a complete test set
allpaths = ["../DeepLoc_full_datasets/datasets/Chong_test_set.hdf5",
            "../DeepLoc_full_datasets/datasets/Chong_valid_set.hdf5",
            "../DeepLoc_full_datasets/datasets/Chong_train_set.hdf5"]

# Path to save test set images to
outdir = "../test_set/"

if not os.path.exists(outdir):
    os.mkdir(outdir)

outpath = outdir + "images/"

# Also extract the CP features in this dataset for benchmarking
featpath = outdir + "cellprofiler_features.txt"

if not os.path.exists(outpath):
    os.mkdir(outpath)

imagedict = dd.io.load(path)

# Extract the cell profiler features
feat = open(featpath, "w")
feat.write("CellID" + "\t")
for header in imagedict['Info1_columns']:
    feat.write(header.decode() + "\t")
feat.write("\n")
feat.close()

# Iterate over all hdf5 files and load them into the single cell output directory
writeindex = 0
for path in allpaths:
    imagedict = dd.io.load(path)
    index = 0
    for image in imagedict['data1']:
        classID = np.where(imagedict['Index1'][index] == 1)[0][0]
        classname = imagedict['Index1_columns'][classID].decode()
        image = np.reshape(image, (2, 64, 64))

        if not os.path.exists(outpath + classname):
            os.makedirs(outpath + classname)

        gfp = Image.fromarray(image[0, :, :])
        rfp = Image.fromarray(image[1, :, :])

        gfp.save(outpath + classname + "/" + classname + "_" + str(writeindex) + "_gfp.tif")
        rfp.save(outpath + classname + "/" + classname + "_" + str(writeindex) + "_rfp.tif")

        feat = open(featpath, "a")
        feat.write(classname + "_" + str(writeindex) + "\t")
        for feature in imagedict['Info1'][index]:
            feat.write(str(feature) + "\t")
        feat.write("\n")
        feat.close()

        index += 1
        writeindex += 1
