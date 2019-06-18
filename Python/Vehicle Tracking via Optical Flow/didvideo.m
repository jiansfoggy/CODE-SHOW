% record video from file ttoh_p
video = VideoWriter('optical_flow.avi'); %create the video object
video.FrameRate = 10;
open(video); %open the file for writing
for i=1:913 %where N is the number of images
  xbt = sprintf('./grab/img_%d.jpg',i);
  I = imread(xbt); %read the next image
  writeVideo(video,I); %write the image to file
end
close(video);