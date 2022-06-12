#==========#
# Packages #
#==========#

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import os, cv2, sys, json, random, shutil
from tf_augmnetation import TFAugmentation

#===========================#
# Train and Valid File Path #
#===========================#

root      = "/your_root_path/ILSVRC/"
train_img = "Data/CLS-LOC/train/"
train_cls = "ImageSets/CLS-LOC/train_cls.txt"
train_map = "devkit/data/map_clsloc.txt"

valid_img = "Data/CLS-LOC/val/"
valid_cls = "ImageSets/CLS-LOC/val.txt"
valid_grd = "devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt"
valid_blk = "devkit/data/ILSVRC2015_clsloc_validation_blacklist.txt"

#===========================#
# Train and Valid Preparion #
#===========================#

class CustomDataset:
    def train_prepare1(self, root, train_img, train_cls, train_map):
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
        lines2, this, folder_name, cls_dict = None, None, None, None
        del folder_name
        del cls_dict
        del lines2
        del this
        print("====The number of training images is ",len(img_path),".")
        print("====The number of training label is ",len(img_lbl),".")
        #The number of training images is  1281167 .
        #The number of training label is  1281167 .
        #img_path = np.array(img_path)
        #img_lbl  = np.array(img_lbl)
        #img_path, img_lbl = shuffle(img_path, img_lbl)
        return img_path, img_lbl

    def valid_prepare1(self, root, valid_img, valid_cls, valid_grd, valid_blk):
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
        #img_path = np.array(img_path)
        #img_lbl  = np.array(img_lbl)
        #img_path, img_lbl = shuffle(img_path, img_lbl)
        return img_path, img_lbl

    def create_dataset(self, img_path, img_lbl, is_validation=False):
        def get_img(file_name):
            #path = bytes.decode(file_name)
            image_raw = tf.io.read_file(file_name)
            img = tf.image.decode_image(image_raw, channels=3)
            img = tf.cast(img, tf.float32) / 255.0
            '''augmentation'''
            if not (is_validation):# or tf.random.uniform([]) <= 0.5):
                img = self._do_augment(img)
            ''''''
            return img

        def get_lbl(img_lbl):
            path = bytes.decode(img_lbl)
            lbl = np.load(path)
            return lbl

        def wrap_get_img(img_path, img_lbl):
            img = tf.numpy_function(get_img, [img_path], [tf.float32])
            #lbl = tf.numpy_function(get_lbl, [img_lbl], [tf.int64])

            return img, img_lbl

        epoch_size = len(img_path)

        img_path = tf.convert_to_tensor(img_path, dtype=tf.string)
        img_lbl  = tf.convert_to_tensor(img_lbl, dtype=tf.int64) 
        # , dtype=tf.int64

        dataset = tf.data.Dataset.from_tensor_slices((img_path, img_lbl))
        dataset = dataset.shuffle(epoch_size)
        dataset = dataset.map(wrap_get_img, num_parallel_calls=32) \
            .batch(100, drop_remainder=True) \
            .prefetch(10)
        return dataset

    def dataset_generator(self, img_path, img_lbl, is_validation=False):
        def load_image(img_path):
            return tf.keras.preprocessing.image.load_img(
                img_path, target_size=[224,224])

        def image_to_array(image):
            #image = tf.cast(image, tf.float32) / 255.0
            return tf.keras.preprocessing.image.img_to_array(image, dtype=np.float32)
        
        def image_preprocess(image_array):
            load_one = tf.keras.preprocessing.image.img_to_array(image_array, dtype=np.float32)      
            return load_one

        images = []
        for image_path in img_path:
            image = load_image(image_path)
            image_array = image_to_array(image)
            images.append(image_array)

        load_imgs = tf.keras.applications.mobilenet_v2.preprocess_input(images)      
            
        return load_imgs, img_lbl
    
    def train_prepare(self, root, train_img, train_cls, train_map):
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
            cls_dict[this[0]] = this[2]
        f.close()
        lines1, this = None, None
        del lines1
        del this
        print("====Total number of category is ",len(cls_dict),".")
        #Total number of category is  1000 .
        
        new_train = "Data/CLS-LOC/train2/"
        # preserve all img path list and label list
        for l2 in lines2:
            this = l2.split()
            folder_name = this[0].split("/")[0]
            subimg_name = this[0].split("/")[1]
            trgt_dir = root+new_train+cls_dict[folder_name]
            if os.path.isdir(trgt_dir):
                dest = shutil.move(root+train_img+this[0]+".JPEG", trgt_dir+"/"+subimg_name+".JPEG")
            else:
                os.mkdir(trgt_dir)
                dest = shutil.move(root+train_img+this[0]+".JPEG", trgt_dir+"/"+subimg_name+".JPEG")
        
        g.close()
        lines2, this, folder_name, cls_dict, dest = None, None, None, None, None
        del folder_name
        del cls_dict
        del lines2
        del this
        del dest
        #img_path = np.array(img_path)
        #img_lbl  = np.array(img_lbl)
        #img_path, img_lbl = shuffle(img_path, img_lbl)
        print("Done: Training Set")
    
    def valid_prepare(self, root, train_map, valid_img, valid_cls, valid_grd, valid_blk):
        # load files and read contents
        f = open(root+valid_cls,'r')
        g = open(root+valid_grd,'r')
        h = open(root+valid_blk,'r')
        k = open(root+train_map,'r')

        lines1 = f.readlines()
        lines2 = g.readlines()
        lines3 = h.readlines()
        lines4 = k.readlines()

        # create containers
        # create containers
        cls_dict = {}
        blk_ls = []
        img_path = []
        img_lbl  = []

        # preserve all categories
        for l4 in lines4:
            this = l4.split()
            cls_dict[this[1]] = this[2]
        k.close()
        lines4, this = None, None
        del lines4
        del this
        print("====Total number of category is ",len(cls_dict),".")
        #Total number of category is  1000 .

        # preserve all black list
        for line in lines3:
            blk_ls.append(line)
        h.close()
        lines3, this = None, None
        del lines3
        del this
        print("====The number of black list is ",len(blk_ls),".")
        #The number of black list is  1762 .

        # preserve all img path list and label list
        Black_List = root+valid_img+"BlckLst"
        if not os.path.isdir(Black_List):
            os.mkdir(Black_List)
        for line in lines1:
            this  = line.split()
            lbl_i = lines2[int(this[1])-1][:-1]
            lbl_n = cls_dict[lbl_i]
            if lbl_i in blk_ls:
                shutil.move(root+valid_img+this[0]+".JPEG", Black_List+"/"+this[0]+".JPEG")
            else:
                trgt_dir = root+valid_img+lbl_n
                if os.path.isdir(trgt_dir):
                    shutil.move(root+valid_img+this[0]+".JPEG", trgt_dir+"/"+this[0]+".JPEG")
                else:
                    os.mkdir(trgt_dir)
                    shutil.move(root+valid_img+this[0]+".JPEG", trgt_dir+"/"+this[0]+".JPEG")                
        f.close()
        g.close()
        lines1, lines2, this, trgt_dir, lbl_n, lbl_i = None, None, None, None, None, None
        del trgt_dir
        del lines1
        del lines2
        del this
        del lbl_n
        del lbl_i
        print("Done: Validation Set")

    def _do_augment(self, img):
        tf_aug = TFAugmentation()
        img = tf_aug.random_rotate(img, p=0.5)
        # img = tf_aug.random_zoom(img, p=0.7)
        img = tf_aug.crop(img, p=0.3)
        img = tf_aug.flip(img, p=0.3)

        img = tf_aug.random_invert_img(img, p=0.2)
        img = tf_aug.to_gsc(img, p=0.3)
        img = tf_aug.random_quality(img, p=0.5)
        img = tf_aug.color(img, p=0.5)

        return img

#===========#
# Main Body #
#===========#

cds = CustomDataset()

cds.train_prepare(root, train_img, train_cls, train_map)
cds.valid_prepare(root, train_map, valid_img, valid_cls, valid_grd, valid_blk)

print("happy end")
