'''
Move any folders of single cell crops that have fewer than 30 cells contained

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

import os
import shutil

rootdir = "../toy_dataset/images/"  # Directory of folders of single cell crops
outdir = "../filtered_dataset/"     # Directory to move filtered dataset into

# Loop over all folders of single cell crops and remove any with less than 30 cells contained
# It says 60 because each cell is 2 channels (the images are stored separately)
for currdir in os.listdir(rootdir):
    if len(os.listdir(rootdir + currdir)) >= 60:
        print (currdir)
        shutil.copytree(rootdir + currdir, outdir + "images/" + currdir)

