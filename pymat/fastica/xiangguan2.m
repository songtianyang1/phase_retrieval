a = dlmread('1.txt');
b = dlmread('1-1.txt');
%a=[11,1,2];
%b=[-11,-1,-2];
r = corr2(a,b)
%r = corr2(a,b)
%c = xcorr(a,b)
