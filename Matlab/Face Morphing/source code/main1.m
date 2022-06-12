% import images and points files
% the three text files are imported by clicking "Import Data" button.
ted_p = textread('./project3/ted_p.txt')+1;
hillary_p = textread('./project3/hil_p.txt')+1;
tri = textread('./project3/tri.txt')+1;
ted = imread('./project3/ted_cruz.jpg');
hillary = imread('./project3/hillary_clinton.jpg');
fimage = im2double(zeros(size(ted)));

% add points on both images
figure(1);
subplot(1,2,1);imshow(ted);title('Ted')
hold on;
plot(ted_p(:,1),ted_p(:,2), 'r.', 'MarkerSize', 7);
subplot(1,2,2);imshow(hillary);title('Hillary')
hold on;
plot(hillary_p(:,1),hillary_p(:,2), 'b.', 'MarkerSize', 7);
figure(7);
subplot(1,2,1)
hold on;
imshow(ted);triplot(tri, ted_p(:,1)', ted_p(:,2)','r');
title('Ted')
subplot(1,2,2)
hold on;
imshow(hillary);triplot(tri, hillary_p(:,1)', hillary_p(:,2)','b');
title('Hillary')

%{
% convert uint8 image to float64
h_1 = im2double(hillary);
t_1 = im2double(ted);

% Find location of feature points in morphed image
% generate final null image matrix
fm=face_morph_p;

% define the scope of alpha
alpha = [0 0.1 0.3 0.5 0.7 0.9 1];

for m = 1:size(alpha,2)
    fm.value=alpha(m);
    fimg_p = round(fm.linkk(fm.value, ted_p, hillary_p));

    % there are two interpolation methods, 1 for bilinear, 2 for nearest
    % neighbor, please choose one to try
    
    method = 1;
    % final warp
    % plot every combo out
    figure(2);
    for i = 1:size(tri,1)
        fimage = fm.start_morph(alpha(m),tri(i,:),ted_p,hillary_p,fimg_p,t_1,h_1,fimage,method);
        bt = sprintf('Combo image %d with alpha = %8.2f',round(m),alpha(m));
        subplot(3,3,m);imshow(im2uint8(fimage));title(bt)     
    end
    
    method = 2;
    % final warp
    % plot every combo out
    figure(3);
    for i = 1:size(tri,1)
        fimage = fm.start_morph(alpha(m),tri(i,:),ted_p,hillary_p,fimg_p,t_1,h_1,fimage,method);
        bt = sprintf('Combo image %d with alpha = %8.2f',round(m),alpha(m));
        subplot(3,3,m);imshow(im2uint8(fimage));title(bt)
        
    end
    
end

fm=face_morph_r;

% define the scope of alpha
alpha = [0 0.1 0.3 0.5 0.7 0.9 1];

for m = 1:size(alpha,2)
    fm.value=alpha(m);
    fimg_p = round(fm.linkk(fm.value, ted_p, hillary_p));

    % there are two interpolation methods, 1 for bilinear, 2 for nearest
    % neighbor, please choose one to try
    method = 1;
    % final warp
    % plot every combo out
    figure(4);
    for i = 1:size(tri,1)
        fimage = fm.start_morph(alpha(m),tri(i,:),ted_p,hillary_p,fimg_p,t_1,h_1,fimage,method);
        bt = sprintf('Combo image %d with alpha = %8.2f',round(m),alpha(m));
        subplot(3,3,m);imshow(im2uint8(fimage));title(bt)     
    end
    
    method = 2;
    % final warp
    % plot every combo out
    figure(5);
    for i = 1:size(tri,1)
        fimage = fm.start_morph(alpha(m),tri(i,:),ted_p,hillary_p,fimg_p,t_1,h_1,fimage,method);
        bt = sprintf('Combo image %d with alpha = %8.2f',round(m),alpha(m));
        subplot(3,3,m);imshow(im2uint8(fimage));title(bt)
        
    end    
end
%}