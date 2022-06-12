% Start Plot Result and Histogram 
function Plott = Plot(IM, TAR, IM_Hist_Eq, TA_Hist_Eq, IM_Hist_Mat,k)
Plott=1;
figure(k);
% Plot source image and its histogram
subplot(3,3,1);imshow(IM);title('image wait for processing')
subplot(3,3,2);imshow(IM_Hist_Eq);title('histogram equalized image')
whos IM_Hist_Eq
subplot(3,3,3);imhist(IM_Hist_Eq);title('histogram of processed image')
% Plot target image and its histogram
subplot(3,3,4);imshow(TAR);title('target wait for processing')
subplot(3,3,5);imshow(TA_Hist_Eq);title('histogram equalized target')
whos TA_Hist_Eq
subplot(3,3,6);imhist(TA_Hist_Eq);title('histogram of processed target')
% Plot matched image and its histogram
subplot(3,3,7);imshow(IM_Hist_Mat);title('histogram matched image')
whos IM_Hist_Mat
subplot(3,3,9);imhist(IM_Hist_Mat);title('histogram of matched image')
end