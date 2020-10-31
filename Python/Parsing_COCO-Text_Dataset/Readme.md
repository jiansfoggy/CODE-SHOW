# Parsing COCO-Text Dataset

Target: get the following values  
> Part A: img_id, file_name, width, height bbox and transcription  
> Part B: trainset, validset, testset

## Step 1 Download COCO-Text dataset
Click the following link to download COCO-Text annotations 2017 v1.4  
*https://vision.cornell.edu/se3/coco-text-2/*
Then, unzip it to your target directory

Click the following link to download COCO train2014 dataset  
*https://cocodataset.org/#download*
Then, unzip it to your target directory.  
> Note: the size of this file is 13GB, it may need more space.

## Step 2 Select one method to parse it

**[Method 1](https://github.com/jiansfoggy/CODE-SHOW/tree/master/Python/Parsing_COCO-Text_Dataset/Method1)**  
This method returns 2 parts separately.  
Get values in Part A via get_ann.py;  
Get values in Part B via assign_img.py.

**[Method 2]**
This method returns 2 parts at the same time 

## Step 3 Run Code  
Method 1:  
```  
#Part A
CUDA_VISIBLE_DEVICES=2,3 python3 get_ann.py

#Part B
CUDA_VISIBLE_DEVICES=2,3 python3 assign_img.py --ann_path /media/data3/jian/Text_Detection/COCO-Text/train_anns --img_path /media/data3/jian/Text_Detection/COCO-Text/train2014/ --folder2save /media/data3/jian/Text_Detection/COCO-Text/coco_train/

CUDA_VISIBLE_DEVICES=2,3 python3 assign_img.py --ann_path /media/data3/jian/Text_Detection/COCO-Text/test_anns --img_path /media/data3/jian/Text_Detection/COCO-Text/train2014/ --folder2save /media/data3/jian/Text_Detection/COCO-Text/coco_test/

CUDA_VISIBLE_DEVICES=2,3 python3 assign_img.py --ann_path /media/data3/jian/Text_Detection/COCO-Text/val_anns --img_path /media/data3/jian/Text_Detection/COCO-Text/train2014/ --folder2save /media/data3/jian/Text_Detection/COCO-Text/coco_valid/  
```

Method 2:

## Step 4 Output  
Now, there should be six files. Three for new splitted annotations, three for new splitted image sets

Next, you can play this dataset with your models.
