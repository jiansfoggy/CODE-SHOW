classdef face_morph_p

    properties
        value
    end
    
    methods (Static)
        function points = linkk(alpha, file1, file2)
            points = round(file1.*(1 - alpha) + file2.*alpha);
        end
    end
    methods (Static)
        % by referring cv2.boundingRect(), we notice that there are 4
        % outputs, minimum x, minimum y, triangle's width and triangle's
        % height. Hence, we code the function and also output this 4 value. 
        function [Tri_BB, Tri_PP] = BoundingBox(tri_p, face_p)
            % make sure scope
            get_six_p = [face_p(tri_p(1),:);face_p(tri_p(2),:);face_p(tri_p(3),:);];
            
            % most up
            x_up = min(get_six_p(:,1));
            
            % most down
            x_down = max(get_six_p(:,1));
            
            % most left
            y_left = min(get_six_p(:,2));

            % most right
            y_right = max(get_six_p(:,2));
            
            % get width and height
            height = x_down-x_up+1;
            width = y_right-y_left+1;
            
            Tri_BB = [x_up y_left width height x_down y_right];
            Tri_PP = get_six_p;
        end
        
        function tform_value = AffineTform(ori_image,tri_ori,tri_f,f_r,method)
            tform = cp2tform(tri_ori,tri_f,'affine');
            if method == 1
                tform_value = imtransform(ori_image, tform,'bilinear','XData',[f_r(1), f_r(5)],'YData',[f_r(2), f_r(6)],'XYScale', [1,1]);
            elseif method == 2
                tform_value = imtransform(ori_image, tform,'nearest','XData',[f_r(1), f_r(5)],'YData',[f_r(2), f_r(6)],'XYScale', [1,1]);
            else
                disp('Only choose between 1 and 2 ^_^')
            end
        end
        
        function fimage = start_morph(alpha,tri,ted_p,hillary_p,fimg_p,t_1,h_1,fimage,method)
            % Find bounding rectangle for each triangle
            transit = tri;
       
            [f_r,tri_f] = face_morph_p.BoundingBox(transit,fimg_p);
    
            % Offset points by left top corner of the respective rectangles
            f_Rect = [tri_f(1,1)-f_r(1),tri_f(1,2)-f_r(2);tri_f(2,1)-f_r(1),tri_f(2,2)-f_r(2);tri_f(3,1)-f_r(1),tri_f(3,2)-f_r(2);];
            
            % Apply warpImage to small rectangular patches
            t_warp = face_morph_p.AffineTform(t_1, ted_p, fimg_p, f_r, method);
            h_warp = face_morph_p.AffineTform(h_1, hillary_p, fimg_p, f_r, method);
            
            % Alpha blend rectangular patches
            fim_Rect = (1.0 - alpha) * t_warp + (alpha * h_warp);
            
            % define mask for RGB each layer
            mask_temp = repmat(poly2mask(f_Rect(:,1),f_Rect(:,2),f_r(3),f_r(4)),1,1,3);
            mask = repmat(poly2mask(f_Rect(:,1),f_Rect(:,2),f_r(3),f_r(4)),1,1,3);
            
            for k = 1:3
                for x = 1:f_r(3)
                    if mask_temp(x,1,k)==0
                        pd=true;
                    else
                        pd=false;
                    end
                    for y = 1:f_r(4)
                        if pd 
                            if (y+2<=f_r(4)) && (mask_temp(x,y+1,k)==0) && (mask_temp(x,y+2,k)==1)
                                mask(x,y,k) = 1;
                                mask(x,y+1,k) = 1;
                                pd=false;
                            elseif (y+1<=f_r(4)) && (mask_temp(x,y+1,k)==1)
                                mask(x,y,k) = 1;
                                pd=false;
                            end
                        else
                            if (y+2<=f_r(4)) && (mask_temp(x,y+1,k)==0) && (mask_temp(x,y+2,k)==0)
                                mask(x,y+1,k) = 1;
                                mask(x,y+2,k) = 1;
                                break;
                            elseif (y+1<=f_r(4)) && (mask_temp(x,y+1,k)==0)
                                mask(x,y+1,k) = 1;
                                break;
                            end
                        end
                    end
                end
                for y = 1:f_r(4)
                    if mask_temp(1,y,k)==0
                        pd=true;
                    else
                        pd=false;
                    end
                    for x = 1:f_r(3)
                        if pd
                            if (x+2<=f_r(3)) && (mask_temp(x+1,y,k)==0) && (mask_temp(x+2,y,k)==1)
                                mask(x,y,k) = 1;
                                mask(x+1,y,k) = 1;
                                pd=false;
                            elseif (x+1<=f_r(3)) && (mask_temp(x+1,y,k)==1)
                                mask(x,y,k) = 1;
                                pd=false;
                            end
                        else
                            if (x+2<=f_r(3)) && (mask_temp(x+1,y,k)==0) && (mask_temp(x+2,y,k)==0)
                                mask(x+1,y,k) = 1;
                                mask(x+2,y,k) = 1;
                                break;
                            elseif (x+1<=f_r(3)) && (mask_temp(x+1,y,k)==0)
                                mask(x+1,y,k) = 1;
                                break;
                            end
                        end
                    end
                end
            end
            % Copy triangular region of the rectangular patch to the output image
            for q=1:3
                fimage(f_r(2):f_r(6),f_r(1):f_r(5),q) = fimage(f_r(2):f_r(6),f_r(1):f_r(5),q) .* (1 - mask(:,:,q)) + (fim_Rect(:,:,q) .* mask(:,:,q)); 
            end
        end
    end
end