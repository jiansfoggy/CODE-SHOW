#==========#
# Packages #
#==========#

import numpy as np
import os, cv2, sys, json, random
from sklearn.utils import shuffle

#===========================#
# Train and Valid File Path #
#===========================#

root = "/your_root_path/ILSVRC/"
train_img = "Data/CLS-LOC/train/"
train_cls = "ImageSets/CLS-LOC/train_cls.txt"
train_map = "devkit/data/map_clsloc.txt"

valid_img = "Data/CLS-LOC/val/"
valid_cls = "ImageSets/CLS-LOC/val.txt"
valid_grd = "devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt"
valid_blk = "devkit/data/ILSVRC2015_clsloc_validation_blacklist.txt"

#===================================#
# Category for Each Training Folder #
#===================================#

f = open(root+train_map,'r')
lines = f.readlines()

cls_dict = {}
for line in lines:
    this = line.split()
    cls_dict[this[0]] = int(this[1])
f.close()
lines, this = None, None
del lines
del this
print("Total number of category is ",len(cls_dict),".")
#Total number of category is  1000 .

#===========================================#
# Image and Label Path List in Training Set #
#===========================================#

f = open(root+train_cls,'r')
g = open("train_img.txt", 'w')
h = open("train_lbl.txt", 'w')
lines = f.readlines()

img_path = []
img_lbl  = []
for line in lines:
    this = line.split()
    img_path.append(root+train_img+this[0]+".JPEG")
    folder_name = this[0].split("/")[0]
    img_lbl.append(cls_dict[folder_name])
    #g.write(root+train_img+this[0]+".JPEG")
    #g.write('\n')
    #h.write(cls_dict[folder_name])
    #h.write('\n')
f.close()
g.close()
h.close()
lines, this, folder_name = None, None, None
del folder_name
del lines
del this
print("The number of training images is ",len(img_path),".")
print("The number of training label is ",len(img_lbl),".")
#The number of training images is  1281167 .
#The number of training label is  1281167 .
img_path = np.array(img_path)
img_lbl  = np.array(img_lbl)

#===============#
# Get Blacklist #
#===============#

f = open(root+valid_blk,'r')
lines = f.readlines()

blk_ls = []
for line in lines:
    blk_ls.append(int(line))
f.close()
lines, this = None, None
del lines
del this
print("The number of black list is ",len(blk_ls),".")
#The number of black list is  1762 .

#=============================================#
# Image and Label Path List in Validation Set #
#=============================================#

f = open(root+valid_cls,'r')
g = open(root+valid_grd,'r')
h = open("valid_img.txt", 'w')
k = open("valid_lbl.txt", 'w')
lines1 = f.readlines()
lines2 = g.readlines()

img_path = []
img_lbl  = []
for line in lines1:
    this = line.split()
    lbl_i = int(lines2[int(this[1])-1])
    if not (lbl_i in blk_ls):
        img_path.append(root+valid_img+this[0]+".JPEG")
        img_lbl.append(lbl_i)
        #h.write(root+valid_img+this[0]+".JPEG")
        #h.write('\n')
        #k.write(lbl_i)
        #k.write('\n')
f.close()
g.close()
h.close()
k.close()
lines1, lines2, this = None, None, None
del lines1
del lines2
del this
print("The number of validation images is ",len(img_path),".")
print("The number of validation label is ",len(img_lbl),".")
#The number of validation images is  48300 .
#The number of validation label is  48300 .
img_path = np.array(img_path)
img_lbl  = np.array(img_lbl)

#===========================#
# Function for Training Set #
#===========================#

def train_prepare(root, train_img, train_cls, train_map):
    # load files and read contents
    f = open(root+train_map,'r')
    g = open(root+train_cls,'r')
    
    lines1 = f.readlines()
    lines2 = g.readlines()
    
    # create containers
    cls_dict = {}
    img_path = []
    img_lbl  = []
    
    # preserve all categories
    for l1 in lines1:
        this = l1.split()
        cls_dict[this[0]] = int(this[1])
    f.close()
    lines1, this = None, None
    del lines1
    del this
    print("====Total number of category is ",len(cls_dict),".")
    #Total number of category is  1000 .
    
    # preserve all img path list and label list
    for l2 in lines2:
        this = l2.split()
        img_path.append(root+train_img+this[0]+".JPEG")
        folder_name = this[0].split("/")[0]
        img_lbl.append(cls_dict[folder_name])
    g.close()
    lines2, this, folder_name, cls_dict = None, None, None
    del folder_name
    del cls_dict
    del lines2
    del this
    print("====The number of training images is ",len(img_path),".")
    print("====The number of training label is ",len(img_lbl),".")
    #The number of training images is  1281167 .
    #The number of training label is  1281167 .
    img_path = np.array(img_path)
    img_lbl  = np.array(img_lbl)
    img_path, img_lbl = shuffle(img_path, img_lbl)
    return img_path, img_lbl

#=============================#
# Function for Validation Set #
#=============================#

def valid_prepare(root, valid_img, valid_cls, valid_grd, valid_blk):
    # load files and read contents
    f = open(root+valid_cls,'r')
    g = open(root+valid_grd,'r')
    h = open(root+valid_blk,'r')
    
    lines1 = f.readlines()
    lines2 = g.readlines()
    lines3 = h.readlines()

    # create containers
    blk_ls = []
    img_path = []
    img_lbl  = []

    # preserve all black list
    for line in lines3:
        blk_ls.append(int(line))
    h.close()
    lines3, this = None, None
    del lines3
    del this
    print("====The number of black list is ",len(blk_ls),".")
    #The number of black list is  1762 .

    # preserve all img path list and label list
    for line in lines1:
        this = line.split()
        lbl_i = int(lines2[int(this[1])-1])
        if not (lbl_i in blk_ls):
            img_path.append(root+valid_img+this[0]+".JPEG")
            img_lbl.append(lbl_i)
    f.close()
    g.close()
    lines1, lines2, this = None, None, None
    del lines1
    del lines2
    del this
    print("====The number of validation images is ",len(img_path),".")
    print("====The number of validation label is ",len(img_lbl),".")
    #The number of validation images is  48300 .
    #The number of validation label is  48300 .
    img_path = np.array(img_path)
    img_lbl  = np.array(img_lbl)
    img_path, img_lbl = shuffle(img_path, img_lbl)
    return img_path, img_lbl

#=============================#

def train_prepare1(root, train_img, train_cls, train_map):
    # load files and read contents
    f = open(root+train_map,'r')
    g = open(root+train_cls,'r')
    
    lines1 = f.readlines()
    lines2 = g.readlines()
    
    # create containers
    cls_dict = {}
    img_path = []
    img_lbl  = []
    img_ls   = []

    # preserve all categories
    for l1 in lines1:
        this = l1.split()
        cls_dict[this[0]] = int(this[1])
    f.close()
    lines1, this = None, None
    del lines1
    del this
    print("====Total number of category is ",len(cls_dict),".")
    #Total number of category is  1000 .
    
    # preserve all img path list and label list
    for l2 in lines2:
        this = l2.split()
        #img_path.append(root+train_img+this[0]+".JPEG")
        img = cv2.imread(root+train_img+this[0]+".JPEG")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224)).astype('float32')
        img_ls.append(img)
        folder_name = this[0].split("/")[0]
        img_lbl.append(cls_dict[folder_name])
    g.close()
    lines2, this, folder_name, cls_dict, img = None, None, None, None
    del folder_name
    del cls_dict
    del lines2
    del this
    del img
    #print("====The number of training images is ",len(img),".")
    print("====The number of training label is ",len(img_lbl),".")
    #The number of training images is  1281167 .
    #The number of training label is  1281167 .
    return img_ls, img_lbl

def valid_prepare1(root, valid_img, valid_cls, valid_grd, valid_blk):
    # load files and read contents
    f = open(root+valid_cls,'r')
    g = open(root+valid_grd,'r')
    h = open(root+valid_blk,'r')
    
    lines1 = f.readlines()
    lines2 = g.readlines()
    lines3 = h.readlines()

    # create containers
    blk_ls   = []
    img_path = []
    img_lbl  = []
    img_ls   = []

    # preserve all black list
    for line in lines3:
        blk_ls.append(int(line))
    h.close()
    lines3, this = None, None
    del lines3
    del this
    print("====The number of black list is ",len(blk_ls),".")
    #The number of black list is  1762 .

    # preserve all img path list and label list
    for line in lines1:
        this = line.split()
        lbl_i = int(lines2[int(this[1])-1])
        if not (lbl_i in blk_ls):
            #img_path.append(root+valid_img+this[0]+".JPEG")
            img = cv2.imread(root+valid_img+this[0]+".JPEG")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224,224)).astype('float32')
            img_ls.append(img)
            img_lbl.append(lbl_i)
    f.close()
    g.close()
    lines1, lines2, this, img = None, None, None, None
    del lines1
    del lines2
    del this
    del img
    print("====The number of validation images is ",len(img_path),".")
    print("====The number of validation label is ",len(img_lbl),".")
    #The number of validation images is  48300 .
    #The number of validation label is  48300 .
    return img_ls, img_lbl

print("---------line---------")
img_valid, lbl_valid = valid_prepare1(root, valid_img, valid_cls, valid_grd, valid_blk)
print("---------line---------")
img_train, lbl_train = train_prepare1(root, train_img, train_cls, train_map)
print("---------line---------")