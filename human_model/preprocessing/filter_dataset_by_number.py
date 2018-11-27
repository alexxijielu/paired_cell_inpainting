'''
Remove any folders of single cell crops that have fewer than 5 cells contained

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

import os
import shutil

# Directory of folders of single cell crops
rootdir = "../human_protein_atlas_single_cell/"

# Loop over all folders of single cell crops and remove any with less than 5 cells contained
# It says 15 because each cell is 3 channels (the images are stored separately)
for currdir in os.listdir(rootdir):
    if len(os.listdir(rootdir + currdir)) <= 15:
        print (currdir, len(os.listdir(rootdir + currdir)))
        shutil.rmtree(rootdir + currdir)


