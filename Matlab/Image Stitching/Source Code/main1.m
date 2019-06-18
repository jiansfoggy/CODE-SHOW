clc;
clear;
%Research on a Novel Medical Image Non-rigid Registration 
%Method Based on Improved SIFT Algorithm
img_right = imread('mcr.jpg');
img_center = imread('mcc.jpg');

Mosaic1 = concat_a1(img_center,img_right);

Mosaic3 = concat_a2(img_center,img_right);