#Important Note:

To achieve the results reported in our manuscript, the model requires a large amount of training data. We have included pretrained weights for both 
of the models we used to extract features for our manuscript. As our training datasets are large (>200 GB), it was not possible to include the full 
data for reproducing these weights. For the yeast model, we have included a small toy dataset of 5 proteins, but models trained on this data will not 
achieve results comparable to our pretrained model and should only be used to verify our training code's functionality. For the human model, included a
small toy dataset of 3 proteins, and we have alsoincluded scripts to systematically download and crop the training data, as this data is more readily
available to the public.

#System Requirements:
Operating system:
This code was tested on a CentOS Linux 7 server

Our code requires and has been tested with Python 3.6.1, with the following packages:
cudatoolkit 9.0
cudnn 7.1.2
deepdish 0.3.4
keras 2.1.5
h5py 2.7.0
mahotas 1.4.4
numpy 1.13.3
pandas 0.22.3
Pillow 5.1.0
skikit-image 0.13.1
scikit-learn 0.19.0
scipy 1.1.0
tensorboard 1.7.0
tensorflow 1.3.0
tensorflow-gpu 1.7.0

Hardware requirements:
- For training our neural networks, we used a Tesla K40c GPU. 
- During training, image preprocessing is parallelized on the CPU. We used 40 Intel� Xeon� Processor E7-8870 v4 CPU cores. 
- For the human model, we recommend at least 250 GB of available hard drive space, if you choose to download the full dataset. 

=====================
Installation:
=====================
Provided source code will run as is as Python scripts; no installation is required.

=====================
Demo:
=====================
We have provided a small toy dataset for the yeast model for demo purposes. Full reproduction of the results reported in our manuscript requires
downloading of the entire yeast and human datasets we used; we have included instructions and estimated running times in the "Instructions for Use" section. 

To run our code on the small toy dataset:

TRAINING:
1. Change directories to the yeast_model directory
1. Open the "opts.py" file and change the "checkpoint_path" variable to the directory to save the weights of the trained model in. By default, this argument is set to the pretrained weights, so change
this or else these weights will be overwritten by your training run. 
3. Run the training script by issuing command line argument "python train.py". This will train the model and save the weights in the checkpoint_path directory as "model_weights.h5"
The training script will run for 30 epochs; with the toy dataset provided, this should take ~20 minutes with the hardware specs above (note that this will be much slower without a GPU)

EXTRACTING FEATURES:
1. Change directories to the yeast_model directory
2. Open the "opts.py" file and change the "checkpoint_path" variable to the directory to find pretrained weights from. We have provided pretrained weights used in the manuscript, 
so to use these, set the variable to "./pretrained_weights/" 
(The dataset to extract features from can also be changed in opts.py by changing the "data_path" argument; by default, this is set to the toy dataset)
3. Run the evaluation script by issuing command line argument "python eval_layer.py". This will extract features from single cells in the toy dataset. Output files will be saved in the
directory given by the "checkpoint_path" variable; these files will contain all features for all of the single cells at each of the 5 convolutional layers as a tab-delimited file. 
Each cell is saved as a row in this text file, and each column is a feature (with the first column being the file name of the single cell crop). 5 files will be saved, with suffixes
"conv1_1" to "conv5_1", which are in order, features extracted from each convolutional layer in the CNN respectively (we had best results using conv3_1 and conv4_1). With the toy
dataset provided, this step should take less than 5 minutes with the hardware specs above. 

BENCHMARK DATA:
1. Labeled data that we used for the benchmark can be downloaded from this link: http://spidey.ccbr.utoronto.ca/~okraus/DeepLoc_full_datasets.zip
2. To unpackage the archives provided in this zip file, "unpackage_test_set.py" in the preprocessing subfolder of the yeast_model folder by issuing command "python unpackage_test_set.py"
(Note that you may need to change the "path" and "allpaths" variable to point to where the Chong archives are saved in your server)
3. This script will save the files in /yeast_model/test_set/ by default; images will be stored in /yeast_model/test_set/images/, and pre-extracted CellProfiler features will be saved as a 
text file (for benchmark purposes, these features aren't used by our code). 
4. Finally, the features can be extracted for the Chong dataset as detailed above in "Extracting Features". Note that in our benchmarks, we merged the "Spindle" and "Spindle Pole" classes.

=====================
Instructions for Use:
=====================
To reproduce the results comparable to those presented in the manuscript (both in terms of training the model and in extracting features), the full datasets used are required. 
For your convenience, for the human model, we provide scripts to automatically download and preprocess all of the images from the Human Protein Atlas. 
Full yeast data can be arranged on request.

NOTE:
All steps will assume the user is in the "human_model" directory

DOWNLOADING THE RAW IMAGES:
1. To download images from the Human Protein Atlas, we have included a python script under /data_download/download_hpa.py To run, call "python download_hpa.py".
2. Depending on your internet connection, this step will take about a week. (Note that this step will download about 24 GB of data)
3. This script will save images in the "human_protein_atlas" directory under the "human_model" directory by default. 

SEGMENTING THE RAW IMAGES:
1. To convert the raw images into a format appropriate for training and feature extraction, we extract crops around the single cells in these images. This is done by using the nuclear channel,
as the nuclei are well separated and easy to segment using trivial computer vision steps in this dataset. 
2. We have included a python script under /data_download/segment_and_crop_human_atlas.py To run, call "python segment_and_crop_human_atlas.py".
3. This script will save images in the "human_protein_atlas_single_cell" directory under the "human_model" directory by default. This step should take 1-2 days, and will save about 200 GB of data. 

FILTERING THE DATA:
1. Two filters are required: first, removing any proteins that are variable, and second, removing any images with fewer than 5 single cell crops. 
2. To remove variable proteins, we have included a script under /preprocessing/delete_variable.py. To run, call "python delete_variable.py". This will use the proteins listed in "variable_proteins.txt",
which are curated directly from the Human Protein Atlas. (This step is not strictly necessary, as spatially variable proteins compromise about 1% of the data, and intensity variable proteins are corrected
for in our per-patch image normalization, but it helps reduce some noise in the dataset for best results.) 
3. To remove cells with too few proteins, we have included a script under /preprocessing/filter_dataset_by_number.py. To run, call "python filter_dataset_by_number.py". This will remove any directories in the
human_protein_atlas_single_cell directory with fewer than 5 cells. 
4. Both of these steps should take under a minute. 

TRAINING:
1. Open the "opts.py" file and change the "checkpoint_path" variable to the directory to save the weights of the trained model in. By default, this argument is set to the pretrained weights, so change
this or else these weights will be overwritten by your training run. 
(Other options in the opts.py will control the batch size and data directory paths, as labelled)
2. Run the training script by issuing command line argument "python train.py". This will train the model and save the weights in the checkpoint_path directory as "model_weights.h5"
The training script will run for 30 epochs; this will take about 3-4 days with the hardware specs we used (training this model without a GPU is very computationally expensive and likely unfeasible.)

EXTRACTING FEATURES:
1. Open the "opts.py" file and change the "checkpoint_path" variable to the directory to find pretrained weights from. We have provided pretrained weights used in the manuscript, 
so to use these, set the variable to "./pretrained_weights/"; alternatively, you can use your own weights from the previous training step. 
2. Run the evaluation script by issuing command line argument "python eval_layer_all_cells.py". This will extract features from single cells in the "human_protein_atlas_single_cell" directory (note that
you will need to download these images first, as in the "Downloading the Raw Images" and "Segmenting the Raw Images" step; the directory of images to extract features from can be changed by changing the 
"datapath" variable. By default, the script is configured to extract features from the 3rd convolutional layer of our network, but this can be changed by changing the "target_layer" variable in the script. 
(Note that we actually used "conv4_1" for the cluster analysis presented in Figure 4.)
3. This script will save all features inside the checkpoint_path directory, in a "features" subdirectory. A separate text file will be saved for each image, containing all of the single cells features for
that image. This step will take approximately 1 hour, depending on which layer you extract features from and how much data you have, and does not require a GPU. 

===============================
UTILITY FILES FOR REPRODUCTION:
===============================
To help reproduce some of the analyses that we conducted in the manuscript, we have also included some utility scripts and additional data files:

A. EXTRACTING FEATURES FROM VGG16:
1. We have provided a script to extract features from a pretrained VGG16 model under /utility/eval_vgg16.py. To run, issue command "python eval_vgg16.py". This script assumes that single cells are saved
in "human_protein_atlas_single_cell", and will output features to the "vgg16_features" folder. This will output the results of different convolutional layers in the network; we observed that the performance
of the pretrained VGG16 network peaks at "block4_conv1". 
2. We tried three strategies for inputting our images: arbitrarily mapping channels to the required 3-channel input of VGG16, inputting all channels as greyscale images independently and concatencating the
features, and inputting only the protein chanel as a greyscale image. The third strategy resulted in superior results on the benchmark by a large margin, so this is reported in the manuscript. (Most information
about protein subcellular localization should be in the protein channel, and the other channels will likely only contribute noise if the features do not model correlations between the protein channel and
these channels appropriately.) 

B. AVERAGING SINGLE CELL FEATURES BY PROTEIN:
1. Averaged features are required to produce the analysis presented in Figure 4.
2. We have included a script to do this in /utility/average_proteins.py. To run, issue command "python eval_vgg16.py". This script assumes that features have been extracted to /pretrained_weights/features/,
and will produce a tab-delimited text file of averaged features in /pretrained_weights/averaged_features.txt. In this file, the rows are proteins, and the columns are features
3. The features in this file can be clustered with hierarchical agglomerative clustering. We recommend normalizing the features by column prior to clustering. 

