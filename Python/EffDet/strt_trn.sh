#!/bin/bash

#image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]

#CUDA_VISIBLE_DEVICES=1 python3 train.py --image_size 512 --batch_size 16 --num_epochs 3 --data_path /media/data3/jian/Text_Detection/YOLOV3/ --log_path ./log
 
#CUDA_VISIBLE_DEVICES=1,2 python3 train.py --image_size 416 --batch_size 16 --device 1,2 --num_epochs 3 --data_path /media/data3/jian/Text_Detection/YOLOV3/ --log_path ./log
 
python3 train.py --image_size 512 --batch_size 16 --num_epochs 2 --data_path ./ --log_path ./log
