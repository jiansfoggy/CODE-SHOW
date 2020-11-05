import argparse
import time,math
import coco_text
import numpy as np
import os,sys,cv2,json
from shutil import copy

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

def add_num(opt):
    class_names = load_classes(opt.cls_path)
    print("The number of class is: ", class_names)
    # 36239
    for root,dirs,files in os.walk(opt.ann_path):
        for fl in files:  
        	anns = open(opt.ann_path+"/"+fl,"r")
        	lbls = open(opt.dst_path+fl,"a")
        	lines = anns.readlines()
        	for line in lines:
        		content = line.strip().split(" ")
        		if (content[0]!="9999"):
        			for i in range(len(class_names)):
        				if content[0]==class_names[i]:
        					content[0] = i
        					lbls.writelines(content)
            lbls.close()
    return print("OK")

def sub_cls(opt):
    class_names = load_classes(opt.cls_path)
    print("The number of class is: ", class_names)
    # 36239
    clas = open(opt.cla_path,"a")
    for root,dirs,files in os.walk(opt.ann_path):
        for fl in files:  
            anns = open(opt.ann_path+"/"+fl,"r")
            lines = anns.readlines()
            for line in lines:
                content = line.strip().split(" ")
                wt_add = int(content[0])
                clas.writelines(class_names[wt_add]+"\n")

    clas.close()
    return print("OK")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # file directory which is waiting for loading
    parser.add_argument('--task', type=int, required=True, help='1 for add_num, 2 for sub_cls')
    parser.add_argument('--ann_path', required=True, help='path to coco_text annotation files')
    parser.add_argument('--cls_path', required=True, help='path to class file')
    parser.add_argument('--cla_path', required=False, help='path to new class file')
    parser.add_argument('--dst_path', required=False, help='path to save new label file')

    opt = parser.parse_args()
    
    if opt.task==1:add_num(opt)
    elif opt.task==2:sub_cls(opt)
    else: print("only input 1 and 2")

# CUDA_VISIBLE_DEVICES=2,3 python3 gen_labels.py --ann_path /media/data3/jian/Text_Detection/COCO-Text/label/valid_anns/ --cls_path /media/data3/jian/Text_Detection/COCO-Text/classes.names --dst_path ./labels/valid/
# CUDA_VISIBLE_DEVICES=2,3 python3 gen_labels.py --ann_path /media/data3/jian/Text_Detection/COCO-Text/label/train_anns/ --cls_path /media/data3/jian/Text_Detection/COCO-Text/classes.names  --dst_path ./labels/train/

# CUDA_VISIBLE_DEVICES=2,3 python3 gen_labels.py --task 2 --ann_path ./labels/train/ --cls_path ./classes.names --cla_path ./new_cls.names
# CUDA_VISIBLE_DEVICES=2,3 python3 gen_labels.py --task 2 --ann_path ./labels/valid/ --cls_path ./classes.names --cla_path ./new_cls.names
