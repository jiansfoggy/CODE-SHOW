import argparse
import time,math
import os,sys,cv2
from shutil import copy

# this code is to split the dataset into three folders, train, test, valid based on the description of annotation json files 
def assign_img(opt):
    print("Start image copying")
	for root,dirs,files in os.walk(opt.ann_path):
        total_img_num = len(files)
        cnt=0
        for fl in files:
        	temp = fl[:-4]+"jpg"
            img_path = opt.img_path+temp            
        	if os.path.isfile(img_path):
                cnt+=1
                # start transferring the images
        		try:
                    copy(img_path,opt.folder2save)
                except IOError as e:
                    print("Unable to copy file. %s" % temp)
                except:
                    print("Unexpected error:", sys.exc_info())

    if cnt==total_img_num: output = "Copying Finished Successfully!"
    elif cnt==0: output = "No one get copied."
    else: output = "Some images are not copied." 

    return print(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # file directory which is waiting for loading
    parser.add_argument('--ann_path', required=True, help='path to train, val, test annotation folder which contains the name of images')
    parser.add_argument('--img_path', required=True, help='path to train, val, test image folders')
    parser.add_argument('--folder2save', required=True, help='path to save train, val, test images')

    opt = parser.parse_args()

    assign_img(opt)

# CUDA_VISIBLE_DEVICES=2,3 python3 assign_img.py --ann_path /media/data3/jian/Text_Detection/COCO-Text/train_anns --img_path /media/data3/jian/Text_Detection/COCO-Text/train2014/ --folder2save /media/data3/jian/Text_Detection/COCO-Text/coco_train/

# CUDA_VISIBLE_DEVICES=2,3 python3 assign_img.py --ann_path /media/data3/jian/Text_Detection/COCO-Text/test_anns --img_path /media/data3/jian/Text_Detection/COCO-Text/train2014/ --folder2save /media/data3/jian/Text_Detection/COCO-Text/coco_test/

# CUDA_VISIBLE_DEVICES=2,3 python3 assign_img.py --ann_path /media/data3/jian/Text_Detection/COCO-Text/val_anns --img_path /media/data3/jian/Text_Detection/COCO-Text/train2014/ --folder2save /media/data3/jian/Text_Detection/COCO-Text/coco_valid/
