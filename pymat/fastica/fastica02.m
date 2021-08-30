
clear;
close all

ms1=imread( 'hunhe02_a1.jpg' );
ms2=imread( 'hunhe02_a2.jpg' );

% ms1=imresize( ms1,[800,800]);
% ms2=imresize( ms2,[800,800]);


aaa=size(ms1)
ms1=reshape(ms1,[],1);%��
ms2=reshape(ms2,[],1);



ms1=double(ms1);
ms2=double(ms2);



a=size(ms1)
mixedsig=[ms1,ms2]; %����ź�Ϊһ�� 3*65535 �ľ���

mixedsig=mixedsig';
% ms1=reshape(ms1,[800,800,3]);
% ms2=reshape(ms2,[800,800,3]);


ms1=reshape(ms1,[1440,1440,3]);
ms2=reshape(ms2,[1440,1440,3]);


%��ms1��ms2�� ms3��������ת�����޷���������
MI1=uint8 (round(ms1)); 
MI2=uint8 (round(ms2));

%bbb=size(MI1)
%��ʾ���ͼ��
figure(2)
subplot(121),imshow(MI1),title( ' ���ͼƬ 1' );
subplot(122),imshow(MI2),title( ' ���ͼƬ 2' );





mixeds_bak=mixedsig; %����Ϻ�����ݱ��ݣ��Ա��ڻָ�ʱֱ�ӵ���
mixeds_mean=zeros(2,1); %��׼��
for i=1:2
 mixeds_mean(i)=mean(mixedsig(i,:));
end %���� mixedsig �ľ�ֵ�ͷ���
for i=1:2
for j=1:size(mixedsig,2)
 mixedsig(i,j)=mixedsig(i,j)-mixeds_mean(i);
end

end %�׻�
mixeds_cov=cov(mixedsig'); %cov Ϊ��Э����ĺ���
[E,D]=eig(mixeds_cov); %��ͼƬ�����Э�������������ֵ�ֽ�
Q=inv(sqrt(D))*(E)'; %QΪ�׻�����
mixeds_white=Q*mixedsig; %mixeds_white Ϊ�׻����ͼƬ����
I=cov(mixeds_white'); %IӦΪ��λ��
%%%%%%%%%%%%%%%%%%FastICA �㷨 %%%%%%%%%%%%%%%%%%%%%%%%%%
X=mixeds_white; %���¶� X���в���
[variablenum,samplenum]=size(X);
numofIC=variablenum; %�ڴ�Ӧ���У�����Ԫ�������ڱ�������
B=zeros(numofIC,variablenum); % �� ʼ �� �� �� �� b �� �� �� �� �� ��B=[b1,b2, ,, ,bd]
for r=1:numofIC
 i=1;
 maxiterationsnum=150; %��������������
 b=2*(rand(numofIC,1)-.5); %������� b�ĳ�ֵ
 b=b/norm(b); %��b ��׼��
while i<=maxiterationsnum+1
if i==maxiterationsnum %ѭ����������
 fprintf( '\n �� %d �� �� �� %d �� �� �� �� �� �� �� ��.' ,r,maxiterationsnum);
break ;
end
 bold=b; %��ʼ��ǰһ�� b �ļĴ���
 u=1;
 t=X'*b;
 g=t.^3;
 dg=3*t.^2;
 b=((1-u)*t'*g*b+u*X*g)/samplenum-mean(dg)*b; %���Ĺ�ʽ
 b=b-B*B'*b; %��b������
 b=b/norm(b);
if abs(abs(b'*bold)-1)<1e-9 %�����������
 B(:,r)=b; %������������ b
break ;
end
 i=i+1;
end
end
%���ݸ�ԭ
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
%��ʾ������ͼ��
figure(3)
subplot(121),imshow(II1),title( ' �� �� �� �� ͼ Ƭ1' );%,subplot(322),imhist(II1),title( ' ����ͼƬ 1��ֱ��ͼ ' );

subplot(122),imshow(II2),title( ' �� �� �� �� ͼ Ƭ2' );%,subplot(324),imhist(II2),title( ' ����ͼƬ 2��ֱ��ͼ ' );
% subplot(133),imshow(II3),title( ' �� �� �� �� ͼ Ƭ3' );%,subplot(326),imhist(II3),title( ' ����ͼƬ 3��ֱ��ͼ ' );
% III1=imsubtract(I1,II1);
% III2=imsubtract(I2,II2);
% III3=imsubtract(I3,II3);
%��ʾ������ͼƬ��ԭͼ�Ĳ�ֵͼ�Լ���Ӧ��ֱ��ͼ %
% figure(4)
% subplot(321),imshow(III1),title( ' �� ֵ ͼ Ƭ1' ),subplot(322);%,imhist(III1),title( ' ��ֵͼƬ 1��ֱ��ͼ ' );
% subplot(323),imshow(III2),title( ' �� ֵ ͼ Ƭ2' ),subplot(324);%,imhist(III2),title( ' ��ֵͼƬ 2��ֱ��ͼ ' );
% subplot(325),imshow(III3),title( ' �� ֵ ͼ Ƭ3' ),subplot(326);%,imhist(III3),title( ' ��ֵͼƬ 3��ֱ��ͼ ' );

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
% subplot(321),imshow(img),title( ' �� ֵ ͼ Ƭ1' ),subplot(322);%,imhist(III1),title( ' ��ֵͼƬ 1��ֱ��ͼ ' );
% subplot(322),imshow(imgn),title( ' �� ֵ ͼ Ƭ1' )

B=8;                %����һ�������ö��ٶ�����λ
MAX=2^B-1;          %ͼ���ж��ٻҶȼ�
MES=sum(sum((img-imgn).^2))/(h*w);     %������
PSNR=20*log10(MAX/sqrt(MES))          %��ֵ�����

% B=8;                %����һ�������ö��ٶ�����λ
% MAX=2^B-1;          %ͼ���ж��ٻҶȼ�
% MES=sum(sum((x-II2).^2))/(h*w);     %������
% PSNR=20*log10(MAX/sqrt(MES));           %��ֵ�����
