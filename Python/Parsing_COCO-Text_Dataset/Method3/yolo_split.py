import argparse
import time,math
import coco_text
import numpy as np
import os,sys,cv2,json
from shutil import copy

# this code will finish 4 tasks
# 1 create a new folder called images, and save splitted train. test, valid image set under this directory
# 2 create a new folder called labels, and save annotation information for train set, test set and valid set under this directory
# 3 create 3 text files to save specific images path for each image in train set, test set and valid set.
# 4 create a new file classes.names to save all unique labels here.
# Note: annotations include the index of class, bbox[x,y,width,height]
# Note: this file can serve to train yolo-v3

def get_key(val,cls_dict):
    for key, value in cls_dict.items():
        if (val == value): return key
    return "key does not exist"

def yolo_split(opt):
    # Load COCO_Text.json here
    ct = coco_text.COCO_Text(opt.COCO_path)
    imgCls = [ct.val,ct.test,ct.train]
    split_div = ["valid","test","train"]
   
    now_path = os.getcwd()
    cls_path = now_path+"/classes.names"
    clspath = open(cls_path,"w")
    classes=[]
    cls_dict={}
    cls_num=0
    sp = 0
    for i in imgCls:
        ann_path = now_path+"/labels/"+split_div[sp]
        set_path = now_path+"/images/"+split_div[sp]
        lst_path = now_path+"/"+split_div[sp]+".txt"
        
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
        # count how many images do we select
        cnt=0
        total_img_num = len(imgs)
  
        lstpath = open(lst_path,"a")

        for j in imgs:
            # get image's description
            img = ct.loadImgs(j)[0]
            iId = img['id']
            fName = img['file_name']

            # save annotation for each image
            savepath = ann_path+"/"+fName[:-3]+"txt"

            # remember to extract width, height
            iWidth=img['width']
            iHeight=img['height']

            already_saved_this_img = False
            """    
            # this is for test set
            lstpath.write(set_path+"/"+fName+"\n")

            trg_path = set_path+"/"           
            src_path = opt.img_path+fName
            try:
                copy(src_path, trg_path)
            except IOError as e:
                print("Unable to copy file. %s" % fName)
            except:
                print("Unexpected error:", sys.exc_info())
            """
      
            for k in anns:
                # get ann's description
                ann = ct.loadAnns(k)[0]
               
                if (ann["image_id"]==iId):
                    aBbox = ann["bbox"]
                   
                    if "utf8_string" in ann:
                        aTran = ann["utf8_string"].strip()
                        saveAnn = open(savepath,"a")
                        
                        # start write 
                        if aTran not in classes:
                            classes.append(aTran)
                            clspath.write(aTran+"\n")
                            cls_dict[cls_num]=aTran
                            #max_val = 640
                            #saveAnn.write(str(cls_num)+" "+str(aBbox[0]/max_val)+" "+str(aBbox[1]/max_val)+" "+str(aBbox[2]/max_val)+" "+str(aBbox[3]/max_val)+"\n")
                            saveAnn.write(str(cls_num)+" "+str(aBbox[0]/iWidth)+" "+str(aBbox[1]/iHeight)+" "+str(aBbox[2]/iWidth)+" "+str(aBbox[3]/iHeight)+"\n")
                            if cls_num<=0 and cls_num>26239: print(cls_num)
                            cls_num+=1
                        else: 
                            cls_key = get_key(aTran,cls_dict)
                            #saveAnn.write(str(cls_num)+" "+str(aBbox[0]/max_val)+" "+str(aBbox[1]/max_val)+" "+str(aBbox[2]/max_val)+" "+str(aBbox[3]/max_val)+"\n")
                            saveAnn.write(str(cls_key)+" "+str(aBbox[0]/iWidth)+" "+str(aBbox[1]/iHeight)+" "+str(aBbox[2]/iWidth)+" "+str(aBbox[3]/iHeight)+"\n")
                        saveAnn.close()
                        if (not already_saved_this_img):
                            #save image directory and name for different division
                            lstpath.write(set_path+"/"+fName+"\n")
                            
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

                    #elif "utf8_string" not in ann:
                        #saveAnn.write("NULL "+str(aBbox[0])+" "+str(aBbox[1])+" "+str(aBbox[2])+" "+str(aBbox[3])+" "+str(iWidth)+" "+str(iHeight)+"\n")
                        """
                        if (not already_saved_this_img):
                            #save image directory and name for different division
                            lstpath.write(set_path+"/"+fName+"\n")
                            
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
                       """
            #saveAnn.close()
        lstpath.close()

        sp+=1

        if cnt==total_img_num: output = "Copying Finished Successfully!" 
        elif cnt==0: output = "No one get copied."
        else: output = "Copying Finished Successfully! But some images are not copied." 
        print(output)

    clspath.close()
    print("The class number is: ", cls_num) 
    return print("Only One more step to finish")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # file directory which is waiting for loading
    parser.add_argument('--COCO_path', required=True, help='path to load coco_text annotation file')
    parser.add_argument('--img_path', required=True, help='path to train, val, test image folders')

    opt = parser.parse_args()

    yolo_split(opt)

# CUDA_VISIBLE_DEVICES=3,3 python3 yolo_split.py --COCO_path ./COCO_Text.json --img_path ./train2014/
