function output=call_Dcorr(input_img_path, pps)

addpath('funcs');

image = double(imread(input_img_path));

Nr = 50;
Ng = 10;
r = linspace(0,1,Nr);
GPU = 1;

% apodize image edges with a cosine function over 20 pixels
image = apodImRect(image,20);

% compute resolution
figID = 'fast';
if GPU 
    [kcMax,A0] = getDcorr(gpuArray(image),r,Ng,figID); gpuDevice(1);
else
    [kcMax,A0] = getDcorr(image,r,Ng,figID);
end

disp(['kcMax : ',num2str(kcMax),', A0 : ',num2str(A0)]);

output=2*pps/kcMax;

end