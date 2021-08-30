clear;
close all

% %读取和显示图像
% I1=imread( 'a.jpg' );
% I2=imread( 'e.jpg' );
% I3=imread( 'c.jpg' );
% figure(1)
% subplot(131),imshow(I1),title( ' 原 图 片1' );%,subplot(322),imhist(I1),title( ' 原图片 1的直方图 ' );
% subplot(132),imshow(I2),title( ' 原 图 片2' );%,subplot(324),imhist(I2),title( ' 原图片 2的直方图 ' );
% subplot(133),imshow(I3),title( ' 原 图 片3' );%;subplot(326),imhist(I3),title( ' 原图片 3的直方图 ' );
% %对信号进行随机混合得到仿真观测信号
% a=size(I1)
% s1=reshape(I1,1,[]);
% s2=reshape(I2,1,[]);
% s3=reshape(I3,1,[]); %将s1 、s2 、s3 变换成一个 1*65536 的行矩阵
% 
% s=[s1;s2;s3]; %s为一个 3*65536 的矩阵
% sig=double(s); %sig 为一个 double 型的矩阵
% %A=[ 0.950129285147175,0.4859824687093,0.456467665168341;
% % 0.231138513574288,0.891298966148902,0.018503643248224;
% % 0.606842583541787,0.762096833027395,0.821407164295253;]
% A=rand(size(sig,1)); %生成一个大小为 3*3 ，取值在 0-1 之间的随机矩阵

% A=[ 1,1,1;
% 1,1,1;
% 1,1,1;]

ms1=imread( 'a1.jpg' );
ms2=imread( 'a2.jpg' );
ms3=imread( 'a3.jpg' );
ms4=imread( 'a4.jpg' );


ms1=reshape(ms1,[],1);%列
ms2=reshape(ms2,[],1);
ms3=reshape(ms3,[],1);
ms4=reshape(ms4,[],1);

ms1=double(ms1);
ms2=double(ms2);
ms3=double(ms3);
ms4=double(ms4);

a=size(ms1)
mixedsig=[ms1,ms2,ms3,ms4]; %混合信号为一个 3*65535 的矩阵
ms1=reshape(ms1,[100,100,3]);
ms2=reshape(ms2,[100,100,3]);
ms3=reshape(ms3,[100,100,3]);
ms4=reshape(ms4,[100,100,3]);

%将ms1、ms2、 ms3四舍五入转换成无符号整型数
MI1=uint8 (round(ms1)); 
MI2=uint8 (round(ms2));
MI3=uint8 (round(ms3));
MI4=uint8 (round(ms4));

b=size(MI1)
%显示混合图像
figure(2)
subplot(141),imshow(MI1),title( ' 混合图片 1' );
subplot(142),imshow(MI2),title( ' 混合图片 2' );
subplot(143),imshow(MI3),title( ' 混合图片 3' );
subplot(144),imshow(MI4),title( ' 混合图片 3' );
mixeds_bak=mixedsig; %将混合后的数据备份，以便在恢复时直接调用
mixeds_mean=zeros(4,1); %标准化
for i=1:4
 mixeds_mean(i)=mean(mixedsig(i,:));
end %计算 mixedsig 的均值和方差
for i=1:4
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
Is1=reshape(ICAeds(1,:),[100,100,3]);
Is2=reshape(ICAeds(2,:),[100,100,3]);
Is3=reshape(ICAeds(3,:),[100,100,3]);
Is4=reshape(ICAeds(4,:),[100,100,3]);

II1=uint8(round(Is1));
II2=uint8(round(Is2));
II3=uint8(round(Is3));
II4=uint8(round(Is4));
%显示分离后的图像
figure(3)
subplot(141),imshow(II1),title( ' 分 离 出 的 图 片1' );%,subplot(322),imhist(II1),title( ' 分离图片 1的直方图 ' );
subplot(142),imshow(II2),title( ' 分 离 出 的 图 片2' );%,subplot(324),imhist(II2),title( ' 分离图片 2的直方图 ' );
subplot(143),imshow(II3),title( ' 分 离 出 的 图 片3' );%,subplot(326),imhist(II3),title( ' 分离图片 3的直方图 ' );
subplot(144),imshow(II4),title( ' 分 离 出 的 图 片3' );

III1=imsubtract(I1,II1);
III2=imsubtract(I2,II2);
III3=imsubtract(I3,II3);
III4=imsubtract(I4,II4);
%显示分离后的图片与原图的差值图以及对应的直方图 %
% figure(4)
% subplot(321),imshow(III1),title( ' 差 值 图 片1' ),subplot(322);%,imhist(III1),title( ' 差值图片 1的直方图 ' );
% subplot(323),imshow(III2),title( ' 差 值 图 片2' ),subplot(324);%,imhist(III2),title( ' 差值图片 2的直方图 ' );
% subplot(325),imshow(III3),title( ' 差 值 图 片3' ),subplot(326);%,imhist(III3),title( ' 差值图片 3的直方图 ' );
