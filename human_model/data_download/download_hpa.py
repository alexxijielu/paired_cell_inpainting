'''
A script to download jpeg images from the Human Protein Atlas, for a given list of proteins to download images for.

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

import urllib.request
import os
import csv
import numpy as np
import time

# URL of the Human Protein Atlas
baselink = "https://www.proteinatlas.org/"
suffix = ".xml"

# Open the list of proteins to download
proteinfile = "./proteinlist.txt"
proteins = open(proteinfile)
proteinlist = csv.reader(proteins, delimiter='\t')
matrix = np.array([row for row in proteinlist])

# Specify the directory to download images into
outdir = "../human_protein_atlas/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

# Download all images
for i in range(1, len(matrix)):
    try:
        start_time = time.time()
        xmllabel = matrix[i][0]
        proteinname = matrix[i][1].replace(".", "-")
        print("Working on ", proteinname)

        link = baselink + xmllabel + suffix
        data = urllib.request.urlopen(link)

        recorded = []
        recording = False
        for line in data:
            if recording:
                recorded.append(line)

            if 'subAssay type="human"' in line.decode():
                recording = True

            if '</subAssay>' in line.decode():
                recording = False

        for line in recorded:
            if "imageUrl" in line.decode() and "blue" in line.decode():
                imageurl = line.decode().split(">")[1].split("<")[0]

                if not os.path.exists(outdir + proteinname + "/"):
                    os.mkdir(outdir + proteinname + "/")

                imagename = imageurl.rsplit("/", 1)[1]
                urllib.request.urlretrieve(imageurl, outdir + proteinname + "/" + imagename)
        print ("Finished in ", time.time() - start_time)
    except:
        proteinname = matrix[i][1]
        print ("Couldn't get file, ", proteinname)
