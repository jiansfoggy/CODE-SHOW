% Start Histogram Equalization
function IM_Hist_Eq = Hist_Eq(IM,sr,M,N)
IM_Hist_E = zeros(M,N);
    for i = 1:M
        for j = 1:N  
            % Limit the pixel value within [0,255]
            if sr(sr(:,1)==IM(i,j),2)<0
                IM_Hist_E(i,j)=0;                
            elseif sr(sr(:,1)==IM(i,j),2)>255
                IM_Hist_E(i,j)=255;               
            else IM_Hist_E(i,j)=sr(sr(:,1)==IM(i,j),2);
            end
        end
    end
% Convert float64 to uint8
IM_Hist_Eq = im2uint8(IM_Hist_E/255);
end