import argparse
import time,math
import coco_text
import numpy as np
import os,sys,cv2,json
from shutil import copy

# extract img_id, file_name, width, height bbox and transcription from loaded json file, COCO_Text.json
# and same them into three different files, train_anns.json, test_anns.json, valid_anns.json
# and split the dataset into train, valid and test folder at the same time

def whole_split(opt):
    imgCls = [ct.train,ct.val,ct.test]
    split_div = ["train","val","test"]
    # Load COCO_Text.json here
    ct = coco_text.COCO_Text(opt.COCO_path)
    sp = 0
    for i in imgCls:
        now_path = os.getcwd()
        ann_path = now_path+"/"+split_div[sp]+"_anns"
        set_path = now_path+"/coco_"+split_div[sp]
        # Start creating new folders to save annotations and images
        try:
            os.mkdir(ann_path)
        except OSError:
            print ("Creation of the directory %s failed" % ann_path)
        else:
            print ("Successfully created the directory %s " % ann_path)
        try:
            os.mkdir(set_path)
        except OSError:
            print ("Creation of the directory %s failed" % set_path)
        else:
            print ("Successfully created the directory %s " % set_path)
        # Load image ids for different set here
        imgs = ct.getImgIds(imgIds=i)
        anns = ct.getAnnIds(imgIds=i)
        
        cnt=0
        total_img_num = len(imgs)
        for j in imgs:
            ind = 0
            img_info = {}
            ind_info = {}
            img = ct.loadImgs(j)[0]
            fName = img['file_name']
            iId = img['id']
            img_info["img_id"]=iId
            img_info["file_name"]=fName
            # remember to extract width, height
            #img_info["width"]=img['width']
            #img_info["height"]=img['height']

            already_saved_this_img = False
            have_text = False

            for k in anns:
                ann = ct.loadAnns(k)[0]
                if (ann["image_id"]==iId):
                    aBbox = ann["bbox"]
                    img_info["bbox"]=aBbox
                    if "utf8_string" in ann:
                        have_text = True
                        aTran = ann["utf8_string"]
                        img_info["transcription"]=aTran
                        ind_info[ind] = img_info
                        ind+=1
                        if (not already_saved_this_img): 
                            trg_path = set_path+"/"
                            src_path = opt.img_path+fName
                            try:
                                copy(src_path, trg_path)
                            except IOError as e:
                                print("Unable to copy file. %s" % fName)
                            except:
                                print("Unexpected error:", sys.exc_info())
                            cnt+=1
                            already_saved_this_img = True

            if have_text:
                savepath=ann_path+"/"+fName[:-3]+"json"
                saveAnn = open(savepath,"w")
                json.dump(ind_info,saveAnn)
                saveAnn.close()
        sp+=1

        if cnt==total_img_num: output = "Copying Finished Successfully!"
        elif cnt==0: output = "No one get copied."
        else: output = "Copying Finished Successfully! But some images are not copied." 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # file directory which is waiting for loading
    parser.add_argument('--COCO_path', required=True, help='path to load coco_text annotation file')
    parser.add_argument('--img_path', required=True, help='path to train, val, test image folders')

    opt = parser.parse_args()

    whole_split(opt)

# CUDA_VISIBLE_DEVICES=2,3 python3 assign_img.py --COCO_path ./COCO_Text.json --img_path ./train2014/
