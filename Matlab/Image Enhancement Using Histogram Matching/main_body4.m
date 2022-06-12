clc;
clear;
% processing the first self taking image
[IM,M,N,Bit,L] = image_info('selfdark.jpeg','color');
[TAR,m,n,bit,l] = image_info('test2.tif','color');
[hi_IM,pr_IM,s_zero] = probability(IM,M,N);
[hi_TAR,pr_TAR,sr_zero] = probability(TAR,m,n);
sr_IM = cumulative(pr_IM, s_zero, L);
sr_TAR = cumulative(pr_TAR, sr_zero, l);
IM_Hist_Eq = Hist_Eq(IM,sr_IM,M,N);
TA_Hist_Eq = Hist_Eq(TAR,sr_TAR,m,n);
% Start Histogram Matching
IM_Hist_Mat = imhistmatch(IM_Hist_Eq, TA_Hist_Eq);

Plot(IM, TAR, IM_Hist_Eq, TA_Hist_Eq, IM_Hist_Mat,5)