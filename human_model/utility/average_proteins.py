'''
Given text files of single cell features extracted by eval_layer_all_cells.py,
(one text file per image), output averaged features for each protein.

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

import csv
import numpy as np
import glob

listfile = "./conversionlist.txt"                               # List of proteins to look for
outputfile = "../pretrained_weights/averaged_features.txt"      # Output file to write averaged features to
indir = "../pretrained_weights/features/"                       # Directory where single cell files are stored

# Open list of proteins
file = open(listfile)
list = csv.reader(file, delimiter='\t')
proteinlist = np.array([row for row in list])

# Iterate over all proteins
for proteinname in proteinlist:

    # Collect all single cell files corresponding to that protein
    protein = proteinname[1].replace(".", "-")
    ID = proteinname[0]
    filelist = glob.glob(indir + protein + "_*.txt")
    print (protein, filelist)

    # Collect all single cell features from the single cell files
    allfeatures = []
    for filename in filelist:
        file = open(filename)
        list = csv.reader(file, delimiter='\t')
        matrix = np.array([row for row in list])
        features = matrix[:, 1:-1].astype(np.float32)
        if allfeatures == []:
            allfeatures = features
        else:
            allfeatures = np.vstack((features, allfeatures))

    # Average all of the single cell features and write this to the output file
    if allfeatures != []:
        average_features = np.nanmean(allfeatures, axis=0)
        output = open(outputfile, "a")
        output.write(ID + "\t")
        for feature in average_features:
            output.write(str(feature) + "\t")
        output.write("\n")