import numpy as np 
from scipy.misc import imread 
import sys, os

kernel = lambda x,y : (np.dot(x,y))


test_dir = sys.argv[1]
test_files = [file for file in os.listdir(test_dir) if file[-3:]=="jpg"]

AtoB = [file for file in test_files if "AtoB" in file]
BtoA = [file for file in test_files if "BtoA" in file]

prefix = "Dataset/facades/"

# AtoB : Phase
KMMD = 0 
items = 0

prefix = "datasets/facades/testA/"

for x_g in AtoB:
    x_r = x_g.split("_")[1]
    im_g = imread(test_dir+"/"+x_g).astype("int64")
    im_r = imread(prefix+x_r).astype("int64")
    for x_g_bis in AtoB:
        if x_g != x_g_bis :
            x_r_bis = x_g_bis.split("_")[1]
            im_g_bis = imread(test_dir+"/"+x_g_bis).astype("int64")
            im_r_bis = imread(prefix+x_r_bis).astype("int64")

            KMMD += kernel(im_r.flatten(),im_r_bis.flatten()) - kernel(im_r.flatten(),im_g.flatten()) -\
                    kernel(im_r_bis.flatten(),im_g_bis.flatten()) + kernel(im_g.flatten(),im_g_bis.flatten())
            items += 1

print("KMMD for AtoB : ",np.sqrt(KMMD/items),"|",KMMD/items)  
# BtoA : Phase
KMMD = 0 
items = 0

prefix = "datasets/facades/testB/"

for x_g in BtoA:
    x_r = x_g.split("_")[1]
    im_g = imread(test_dir+"/"+x_g).astype("int64")
    im_r =  imread(prefix+x_r).astype("int64")
    for x_g_bis in BtoA:
        if x_g != x_g_bis :
                
            x_r_bis = x_g_bis.split("_")[1]
            im_g_bis = imread(test_dir+"/"+x_g_bis).astype("int64")
            im_r_bis = imread(prefix+x_r_bis).astype("int64")

            KMMD += kernel(im_r.flatten(),im_r_bis.flatten()) - 2*kernel(im_r.flatten(),im_g.flatten()) \
            + kernel(im_g.flatten(),im_g_bis.flatten())
            items += 1

print("KMMD for BtoA : ",np.sqrt(KMMD/items),"|",KMMD/items)  