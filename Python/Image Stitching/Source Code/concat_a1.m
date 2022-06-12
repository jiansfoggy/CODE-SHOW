function mosaic = concat_a1(img_center,img_right)

% convert the image to grayscale
gray_center = im2single(rgb2gray(img_center));
gray_right = im2single(rgb2gray(img_right));

% get key points in harris corner detector and
% get descriptor in SIFT
[kp_c,des_c] = vl_covdet(gray_center, 'Method', 'HarrisLaplace');
[kp_r,des_r] = vl_covdet(gray_right, 'Method', 'HarrisLaplace');

%show key points and descriptors
figure(1);
subplot(1,2,1);imshow(img_center);
h_c = vl_plotframe(kp_c(:,1000:3000));set(h_c,'color','y','linewidth',2);
s_c = vl_plotsiftdescriptor(des_c(:,1000:3000),kp_c(:,1000:3000));set(s_c,'color','g');
title('Feature Point for Center Image')
subplot(1,2,2);imshow(img_right);
h_r = vl_plotframe(kp_r(:,1000:3000));set(h_r,'color','y','linewidth',2);
s_r = vl_plotsiftdescriptor(des_r(:,1000:3000),kp_r(:,1000:3000));set(s_r,'color','g');
title('Feature Point for Right Image')

% Get Best Matching
[match_rc, dist_rc] = vl_ubcmatch(des_c, des_r);

% Implement RANSAC to get Homography Parameter and Inlier Index
index1=match_rc(1,:);
homogpnt_c=kp_c(1:2,index1);
index2=match_rc(2,:);
homogpnt_r=kp_r(1:2,index2);
[H, inliers_index]=ransacfithomography(homogpnt_c,homogpnt_r,.005);

dh1 = max(size(img_right,1)-size(img_center,1),0) ;
dh2 = max(size(img_center,1)-size(img_right,1),0) ;

% Plot the Original Matching and Best Matching Out
figure(2) ; clf ;
subplot(2,1,1) ;
imagesc([padarray(img_center,dh1,'post') padarray(img_right,dh2,'post')]) ;
o = size(img_center,2) ;
line([kp_c(1,match_rc(1,:));kp_r(1,match_rc(2,:))+o], ...
     [kp_c(2,match_rc(1,:));kp_r(2,match_rc(2,:))]) ;
numMatches = size(match_rc,2);
title(sprintf('%d tentative matches', numMatches)) ;
axis image off ;

subplot(2,1,2) ;
imagesc([padarray(img_center,dh1,'post') padarray(img_right,dh2,'post')]) ;
o = size(img_center,2) ;
line([kp_c(1,match_rc(1,inliers_index));kp_r(1,match_rc(2,inliers_index))+o], ...
     [kp_c(2,match_rc(1,inliers_index));kp_r(2,match_rc(2,inliers_index))]) ;
title(sprintf('%d (%.2f%%) inliner matches out of %d', ...
              size(inliers_index,2), ...
              100*size(inliers_index,2)/numMatches, ...
              numMatches)) ;
axis image off ;
drawnow ;

% Do Backward Warping
box2 = [1  size(img_right,2) size(img_right,2)  1 ;
        1  1           size(img_right,1)  size(img_right,1) ;
        1  1           1            1 ] ;
box2_ = H\box2 ;
box2_(1,:) = box2_(1,:) ./ box2_(3,:) ;
box2_(2,:) = box2_(2,:) ./ box2_(3,:) ;
ur = min([1 box2_(1,:)]):max([size(img_center,2) box2_(1,:)]) ;
vr = min([1 box2_(2,:)]):max([size(img_center,1) box2_(2,:)]) ;

[u,v] = meshgrid(ur,vr) ;
img_c = vl_imwbackward(im2double(img_center),u,v) ;

z_ = H(3,1) * u + H(3,2) * v + H(3,3) ;
u_ = (H(1,1) * u + H(1,2) * v + H(1,3)) ./ z_ ;
v_ = (H(2,1) * u + H(2,2) * v + H(2,3)) ./ z_ ;
img_r = vl_imwbackward(im2double(img_right),u_,v_) ;

% Blend First Image with Warped one and Plot it Out
mass = ~isnan(img_c) + ~isnan(img_r) ;
img_c(isnan(img_c)) = 0 ;
img_r(isnan(img_r)) = 0 ;
mosaic = (img_c + img_r) ./ mass ;

figure(3) ; clf ;
imagesc(mosaic) ; axis image off ;
title('Mosaic Left and Right Image') ;
end