# Attetion 1

yolo_split.py is the final version for parsing coco-text.  
This code will finish 4 tasks  
1. create a new folder called images, and save splitted train. test, valid image set under this directory.  
2. create a new folder called labels, and save annotation information for train set, test set and valid set under this directory.  
3. create 3 text files to save specific images path for each image in train set, test set and valid set.  
4. create a new file classes.names to save all unique labels here.  
* Note: annotations include the index of class, bbox[x,y,width,height]  
* Note: this file can serve to train yolo-v3  

# Attention 2

add_num() function in sub_class.py is to substitude the word transcription into class index from the label files gotten from the method 2.

sub_cls() function in sub_class.py is to create a small subset for class file.
