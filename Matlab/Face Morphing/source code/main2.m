% import images and points files
% the three text files are imported by clicking "Import Data" button.
ted_p = textread('./project3/ted_p.txt')+1;
hillary_p = textread('./project3/hil_p.txt')+1;
tri = textread('./project3/tri.txt')+1;
ted = imread('./project3/ted_cruz.jpg');
hillary = imread('./project3/hillary_clinton.jpg');
fimage = zeros(int64(size(ted)));

% convert uint8 image to float64
h_1 = im2double(hillary);
t_1 = im2double(ted);

% Find location of feature points in morphed image
% generate final null image matrix
fm_p=face_morph_p;

% preparing 101 images to record video
% aplha from 0 to 1, the distance is 0.01
% define the scope of alpha then divide 100
image_num = sort(randperm(101)-1);

% there are two interpolation methods, 1 for bilinear, 2 for nearest
% neighbor, please choose one to try
method = 2;

% save image here
workingDir='ttoh_p';
mkdir(workingDir,'ttoh_p')

for m = 1:size(image_num,2)
    fm_p.value=image_num(m)/100;
    fimg_p = round(fm_p.linkk(fm_p.value, ted_p, hillary_p));

    % final warp
    % plot every combo out
    figure(6);
    for i = 1:size(tri,1)
        fimage = fm_p.start_morph(image_num(m)/100,tri(i,:),ted_p,hillary_p,fimg_p,t_1,h_1,fimage,method);
    end
    bt = sprintf('Combo image %d with alpha = %8.2f',round(m),image_num(m)/100);
    imshow(im2uint8(fimage));title(bt);
    xbt = sprintf('./ttoh_p/%d.jpeg',m);
    saveas(gcf,xbt)
end

% Find location of feature points in morphed image
% generate final null image matrix
fm_r=face_morph_r;

% preparing 101 images to record video
% aplha from 0 to 1, the distance is 0.01
% define the scope of alpha then divide 100
image_num = sort(randperm(101)-1);

% there are two interpolation methods, 1 for bilinear, 2 for nearest
% neighbor, please choose one to try
method = 2;

% save image here
workingDir='ttoh_r';
mkdir(workingDir,'ttoh_r')

for m = 1:size(image_num,2)
    fm_r.value=image_num(m)/100;
    fimg_p = round(fm_r.linkk(fm_r.value, ted_p, hillary_p));

    % final warp
    % plot every combo out
    figure(6);
    for i = 1:size(tri,1)
        fimage = fm_r.start_morph(image_num(m)/100,tri(i,:),ted_p,hillary_p,fimg_p,t_1,h_1,fimage,method);
    end
    bt = sprintf('Combo image %d with alpha = %8.2f',round(m),image_num(m)/100);
    imshow(im2uint8(fimage));title(bt);
    xbt = sprintf('./ttoh_r/%d.jpeg',m);
    saveas(gcf,xbt)
end