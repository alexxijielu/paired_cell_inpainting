'''
Scripts to load and prepare yeast single cells for input into the CNN during training.

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

import os
import numpy as np
from PIL import Image


class Dataset(object):
    '''We store information about the dataset as a class.'''
    def __init__(self):
        self.image_ids = []
        self.image_info = []

    '''Add an individual image to the class. Called iteratively by add_dataset().
    For memory purposes, this class loads only the paths of each image, and returns the 
    actual images as needed.'''
    def add_image(self, image_id, path, name):
        image_info = {
            "id": image_id,         # Unique integer representation of the image
            "path": path,           # Path where the image is stored
            "name": name,           # Name of the image
        }
        self.image_info.append(image_info)

    def add_dataset(self, root_dir):
        i = 0       # Used to assign a unique integer index to each image

        # Iterate over every image in the dataset
        for currdir in os.listdir(root_dir):
            print ("Adding " + currdir)

            # Get all the GFP images only
            all_files = os.listdir(root_dir + currdir)
            image_names = []
            for file in all_files:
                if ("_gfp" in file):
                    image_names.append(file)

            # Add images
            for j in range(len(image_names)):
                self.add_image(
                    image_id=i,
                    path=root_dir + currdir + "/",
                    name=image_names[j])
                i += 1

    '''Load and return the image indexed by the integer given by image_id'''
    def load_image(self, image_id):
        # Get the path of the image
        path = self.image_info[image_id]['path']
        proteinname = self.image_info[image_id]['name']
        brightfieldname = proteinname.replace("_gfp", "_bf")

        # Load and return all two channels for yeast data
        protein = np.array(Image.open(path + proteinname))
        try:
            brightfield = np.array(Image.open(path + brightfieldname))
        except FileNotFoundError:
            brightfieldname = proteinname.replace("_gfp", "_rfp")
            brightfield = np.array(Image.open(path + brightfieldname))

        return protein, brightfield

    '''Load and return the image indexed by the integer given by image_id; also returns the
        name of the image, just for debugging purposes'''
    def load_image_with_label(self, image_id):
        # Get the path of the image
        path = self.image_info[image_id]['path']
        proteinname = self.image_info[image_id]['name']
        brightfieldname = proteinname.replace("_gfp", "_bf")
        label = proteinname.rsplit("_", -1)[0]

        # Load and return all two channels for yeast data, as well as the name of the image
        protein = np.array(Image.open(path + proteinname))
        try:
            brightfield = np.array(Image.open(path + brightfieldname))
        except FileNotFoundError:
            brightfieldname = proteinname.replace("_gfp", "_rfp")
            brightfield = np.array(Image.open(path + brightfieldname))

        return protein, brightfield, label

    '''Sample a pair for the given image by drawing with equal probability from the folder'''
    def sample_pair_equally(self, image_id):
        path = self.image_info[image_id]['path']
        name = self.image_info[image_id]['name']

        # Get all other images in the same path as the image given by image_id
        all_files = os.listdir(path)
        image_names = []
        for file in all_files:
            if ("_gfp" in file and file != name):
                image_names.append(file)

        # If the directory has more than one image, sample a random image and return it
        if len(all_files) > 1:
            sampled_image = np.random.choice(image_names)

            proteinname = sampled_image
            brightfieldname = proteinname.replace("_gfp", "_bf")

            protein = np.array(Image.open(path + proteinname))
            try:
                brightfield = np.array(Image.open(path + brightfieldname))
            except FileNotFoundError:
                brightfieldname = proteinname.replace("_gfp", "_rfp")
                brightfield = np.array(Image.open(path + brightfieldname))


            return protein, brightfield
        else:
            raise ValueError("Directory " + path + " has only one image.")

    '''Prepares the dataset file for use.'''
    def prepare(self):
        # Build (or rebuild) everything else from the info dicts.
        self.num_images = len(self.image_info)
        self.image_ids = np.arange(self.num_images)
