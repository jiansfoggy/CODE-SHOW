img_moa1 = imread('mosaic_rc_a1.png');
img_moa2 = imread('mosaic_rc_a2.png');
img_left = imread('mcl.jpg');

%mosaic2 = concat_a1(img_left,img_moa1);
%mosaic4 = concat_a2(img_left,img_moa2);

Mosaic2 = concat_a1(img_left,Mosaic1);

Mosaic4 = concat_a2(img_left,Mosaic3);