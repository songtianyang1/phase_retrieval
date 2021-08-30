
clear;
close all

ms1=imread( 'hunhe02_a1.jpg' );
ms2=imread( 'hunhe02_a2.jpg' );

% ms1=imresize( ms1,[800,800]);
% ms2=imresize( ms2,[800,800]);


aaa=size(ms1)
ms1=reshape(ms1,[],1);%列
ms2=reshape(ms2,[],1);



ms1=double(ms1);
ms2=double(ms2);



a=size(ms1)
mixedsig=[ms1,ms2]; %混合信号为一个 3*65535 的矩阵

mixedsig=mixedsig';
% ms1=reshape(ms1,[800,800,3]);
% ms2=reshape(ms2,[800,800,3]);


ms1=reshape(ms1,[1440,1440,3]);
ms2=reshape(ms2,[1440,1440,3]);


%将ms1、ms2、 ms3四舍五入转换成无符号整型数
MI1=uint8 (round(ms1)); 
MI2=uint8 (round(ms2));

%bbb=size(MI1)
%显示混合图像
figure(2)
subplot(121),imshow(MI1),title( ' 混合图片 1' );
subplot(122),imshow(MI2),title( ' 混合图片 2' );





mixeds_bak=mixedsig; %将混合后的数据备份，以便在恢复时直接调用
mixeds_mean=zeros(2,1); %标准化
for i=1:2
 mixeds_mean(i)=mean(mixedsig(i,:));
end %计算 mixedsig 的均值和方差
for i=1:2
for j=1:size(mixedsig,2)
 mixedsig(i,j)=mixedsig(i,j)-mixeds_mean(i);
end

end %白化
mixeds_cov=cov(mixedsig'); %cov 为求协方差的函数
[E,D]=eig(mixeds_cov); %对图片矩阵的协方差函数进行特征值分解
Q=inv(sqrt(D))*(E)'; %Q为白化矩阵
mixeds_white=Q*mixedsig; %mixeds_white 为白化后的图片矩阵
I=cov(mixeds_white'); %I应为单位阵
%%%%%%%%%%%%%%%%%%FastICA 算法 %%%%%%%%%%%%%%%%%%%%%%%%%%
X=mixeds_white; %以下对 X进行操作
[variablenum,samplenum]=size(X);
numofIC=variablenum; %在此应用中，独立元个数等于变量个数
B=zeros(numofIC,variablenum); % 初 始 化 列 向 量 b 的 寄 存 矩 阵 ，B=[b1,b2, ,, ,bd]
for r=1:numofIC
 i=1;
 maxiterationsnum=150; %设置最大迭代次数
 b=2*(rand(numofIC,1)-.5); %随机设置 b的初值
 b=b/norm(b); %对b 标准化
while i<=maxiterationsnum+1
if i==maxiterationsnum %循环结束处理
 fprintf( '\n 第 %d 分 量 在 %d 次 迭 代 内 并 不 收 敛.' ,r,maxiterationsnum);
break ;
end
 bold=b; %初始化前一步 b 的寄存器
 u=1;
 t=X'*b;
 g=t.^3;
 dg=3*t.^2;
 b=((1-u)*t'*g*b+u*X*g)/samplenum-mean(dg)*b; %核心公式
 b=b-B*B'*b; %对b正交化
 b=b/norm(b);
if abs(abs(b'*bold)-1)<1e-9 %如果收敛，则
 B(:,r)=b; %保存所得向量 b
break ;
end
 i=i+1;
end
end
%数据复原
ICAeds=B'*Q*mixeds_bak;
ICAeds_bak=ICAeds;
ICAeds=abs(55*ICAeds);
Is1=reshape(ICAeds(1,:),[1440,1440,3]);
Is2=reshape(ICAeds(2,:),[1440,1440,3]);
% Is3=reshape(ICAeds(3,:),[100,100,3]);
II1=uint8(round(Is1));
II2=uint8(round(Is2));
imwrite(II1,'new.jpg','jpg')



% II3=uint8(round(Is3));
%显示分离后的图像
figure(3)
subplot(121),imshow(II1),title( ' 分 离 出 的 图 片1' );%,subplot(322),imhist(II1),title( ' 分离图片 1的直方图 ' );

subplot(122),imshow(II2),title( ' 分 离 出 的 图 片2' );%,subplot(324),imhist(II2),title( ' 分离图片 2的直方图 ' );
% subplot(133),imshow(II3),title( ' 分 离 出 的 图 片3' );%,subplot(326),imhist(II3),title( ' 分离图片 3的直方图 ' );
% III1=imsubtract(I1,II1);
% III2=imsubtract(I2,II2);
% III3=imsubtract(I3,II3);
%显示分离后的图片与原图的差值图以及对应的直方图 %
% figure(4)
% subplot(321),imshow(III1),title( ' 差 值 图 片1' ),subplot(322);%,imhist(III1),title( ' 差值图片 1的直方图 ' );
% subplot(323),imshow(III2),title( ' 差 值 图 片2' ),subplot(324);%,imhist(III2),title( ' 差值图片 2的直方图 ' );
% subplot(325),imshow(III3),title( ' 差 值 图 片3' ),subplot(326);%,imhist(III3),title( ' 差值图片 3的直方图 ' );

% img=imread( 'w1440.jpg' );
% imgn=imread( 'x.jpg' );

img=imread( 'z.jpg' );
imgn=imread( 'new.jpg' );

% imgn=II1;
%imgn=imresize(II1,[100,100]);
aaa=size(img)
bbb=size(II1)

[h w]=size(img);


% imgn=imresize(img,[floor(h/2) floor(w/2)]);
% imgn=imresize(imgn,[h w]);


img=double(w);
imgn=double(II1);
%coef=corr(img,imgn);
figure(4)
subplot(121),imshow(imgn)
% subplot(321),imshow(img),title( ' 差 值 图 片1' ),subplot(322);%,imhist(III1),title( ' 差值图片 1的直方图 ' );
% subplot(322),imshow(imgn),title( ' 差 值 图 片1' )

B=8;                %编码一个像素用多少二进制位
MAX=2^B-1;          %图像有多少灰度级
MES=sum(sum((img-imgn).^2))/(h*w);     %均方差
PSNR=20*log10(MAX/sqrt(MES))          %峰值信噪比

% B=8;                %编码一个像素用多少二进制位
% MAX=2^B-1;          %图像有多少灰度级
% MES=sum(sum((x-II2).^2))/(h*w);     %均方差
% PSNR=20*log10(MAX/sqrt(MES));           %峰值信噪比
