import os,json
import coco_text
import numpy as np

# extract img_id, file_name, bbox and transcription from loaded json file, COCO_Text.json
# and same them into three different files, train_anns.json, test_anns.json, valid_anns.json
ct = coco_text.COCO_Text('COCO_Text.json')
imgCls = [ct.train,ct.val,ct.test]
split_div = ["train","val","test"]
sp = 0
for i in imgCls:
    now_path = os.getcwd()
    ann_path = now_path+"/"+split_div[sp]+"_anns"
    try:
        os.mkdir(ann_path)
    except OSError:
        print ("Creation of the directory %s failed" % ann_path)
    else:
        print ("Successfully created the directory %s " % ann_path)
    imgs = ct.getImgIds(imgIds=i)
    anns = ct.getAnnIds(imgIds=i)
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
        
        for k in anns:
            ann = ct.loadAnns(k)[0]
            if (ann["image_id"]==iId):
                aBbox = ann["bbox"]
                img_info["bbox"]=aBbox
                if "utf8_string" in ann:
                    aTran = ann["utf8_string"]
                    img_info["transcription"]=aTran
                    ind_info[ind] = img_info
                    ind+=1
                else:
                    ind_info[ind] = img_info
                    ind+=1

        savepath=ann_path+"/"+fName[:-3]+"json"
        saveAnn = open(savepath,"w")
        json.dump(ind_info,saveAnn)
        saveAnn.close()
    sp+=1

# CUDA_VISIBLE_DEVICES=2,3 python3 get_ann.py
