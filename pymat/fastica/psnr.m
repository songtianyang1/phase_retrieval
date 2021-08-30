close all;
clear all;
clc;

img=imread('z1440.jpg');
imgn=imread('new.jpg');
[h w]=size(img);
% imgn=imresize(img,[floor(h/2) floor(w/2)]);
% imgn=imresize(imgn,[h w]);
img=double(img);
imgn=double(imgn);

aaa=size(img)
bbb=size(imgn)
img=imresize(img,[1440 1440]);
imgn=imresize(imgn,[1440 1440]);
aaa=size(img)

B=8;                %编码一个像素用多少二进制位
MAX=2^B-1;          %图像有多少灰度级
MES=sum(sum((img-imgn).^2))/(h*w)    %均方差
PSNR=20*log10(MAX/sqrt(MES))          %峰值信噪比

%r = corr2(img,imgn)