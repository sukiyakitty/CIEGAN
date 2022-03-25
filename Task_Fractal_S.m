function output=Task_Fractal_S(input_img_path)
% Only after image acquisition processing!
% Return Only one image fractal

img = imread(input_img_path);
[boxdim,Nboxes,handlefig]=boxdim_binaire(img);
close all;
clear img;
output=boxdim;

end