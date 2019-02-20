% Basic preparion
function [IM,M,N,Bit,L] = image_info(insert_image,tell)
% deal with gray scale image
if tell=="gray"    
    IM=imread(insert_image);
    % convert image to float64 style
    IM=im2double(IM);
    [M,N] = size(IM);
    % get the bit value of image
    IM_in = imfinfo(insert_image);
    Bit=IM_in.BitDepth;
    L=2^Bit;
    
% deal with color image
elseif tell=="color"
    self = imread(insert_image);
    % convert image to gray scale
    IM = rgb2gray(self);
    % convert image to float64 style
    IM=im2double(IM);
    [M,N] = size(IM);
    % get the bit value of image
    IM_in = imfinfo(insert_image);
    Bit=IM_in.BitDepth;
    L=2^(Bit/3);
    
end
end