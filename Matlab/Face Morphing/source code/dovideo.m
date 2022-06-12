% record video from file ttoh_p
video = VideoWriter('ted_to_hillary_p.avi'); %create the video object
video.FrameRate = 10;
open(video); %open the file for writing
for i=1:101 %where N is the number of images
  xbt = sprintf('./ttoh_p/%d.jpeg',i);
  I = imread(xbt); %read the next image
  writeVideo(video,I); %write the image to file
end
close(video);
% record video from file ttoh_r
video = VideoWriter('ted_to_hillary_r.avi'); %create the video object
video.FrameRate = 10;
open(video); %open the file for writing
for i=1:101 %where N is the number of images
  xbt = sprintf('./ttoh_r/%d.jpeg',i);
  I = imread(xbt); %read the next image
  writeVideo(video,I); %write the image to file
end
close(video);            