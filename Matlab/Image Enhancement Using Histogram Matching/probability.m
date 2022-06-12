% Calculate probability of each appearing intensity value r_k
function [hi,pr,s_zero] = probability(IM,M,N)
[counts,inten_value] = imhist(IM);
count=counts/(M*N);    
pr = horzcat(inten_value,count);
hi = horzcat(inten_value,counts);
y = zeros(256,1);
s_zero = horzcat(inten_value,y);
end


