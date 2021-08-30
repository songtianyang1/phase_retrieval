clear;
close all

% %��ȡ����ʾͼ��
% I1=imread( 'a.jpg' );
% I2=imread( 'e.jpg' );
% I3=imread( 'c.jpg' );
% figure(1)
% subplot(131),imshow(I1),title( ' ԭ ͼ Ƭ1' );%,subplot(322),imhist(I1),title( ' ԭͼƬ 1��ֱ��ͼ ' );
% subplot(132),imshow(I2),title( ' ԭ ͼ Ƭ2' );%,subplot(324),imhist(I2),title( ' ԭͼƬ 2��ֱ��ͼ ' );
% subplot(133),imshow(I3),title( ' ԭ ͼ Ƭ3' );%;subplot(326),imhist(I3),title( ' ԭͼƬ 3��ֱ��ͼ ' );
% %���źŽ��������ϵõ�����۲��ź�
% a=size(I1)
% s1=reshape(I1,1,[]);
% s2=reshape(I2,1,[]);
% s3=reshape(I3,1,[]); %��s1 ��s2 ��s3 �任��һ�� 1*65536 ���о���
% 
% s=[s1;s2;s3]; %sΪһ�� 3*65536 �ľ���
% sig=double(s); %sig Ϊһ�� double �͵ľ���
% %A=[ 0.950129285147175,0.4859824687093,0.456467665168341;
% % 0.231138513574288,0.891298966148902,0.018503643248224;
% % 0.606842583541787,0.762096833027395,0.821407164295253;]
% A=rand(size(sig,1)); %����һ����СΪ 3*3 ��ȡֵ�� 0-1 ֮����������

% A=[ 1,1,1;
% 1,1,1;
% 1,1,1;]

ms1=imread( 'a1.jpg' );
ms2=imread( 'a2.jpg' );
ms3=imread( 'a3.jpg' );
ms4=imread( 'a4.jpg' );


ms1=reshape(ms1,[],1);%��
ms2=reshape(ms2,[],1);
ms3=reshape(ms3,[],1);
ms4=reshape(ms4,[],1);

ms1=double(ms1);
ms2=double(ms2);
ms3=double(ms3);
ms4=double(ms4);

a=size(ms1)
mixedsig=[ms1,ms2,ms3,ms4]; %����ź�Ϊһ�� 3*65535 �ľ���
ms1=reshape(ms1,[100,100,3]);
ms2=reshape(ms2,[100,100,3]);
ms3=reshape(ms3,[100,100,3]);
ms4=reshape(ms4,[100,100,3]);

%��ms1��ms2�� ms3��������ת�����޷���������
MI1=uint8 (round(ms1)); 
MI2=uint8 (round(ms2));
MI3=uint8 (round(ms3));
MI4=uint8 (round(ms4));

b=size(MI1)
%��ʾ���ͼ��
figure(2)
subplot(141),imshow(MI1),title( ' ���ͼƬ 1' );
subplot(142),imshow(MI2),title( ' ���ͼƬ 2' );
subplot(143),imshow(MI3),title( ' ���ͼƬ 3' );
subplot(144),imshow(MI4),title( ' ���ͼƬ 3' );
mixeds_bak=mixedsig; %����Ϻ�����ݱ��ݣ��Ա��ڻָ�ʱֱ�ӵ���
mixeds_mean=zeros(4,1); %��׼��
for i=1:4
 mixeds_mean(i)=mean(mixedsig(i,:));
end %���� mixedsig �ľ�ֵ�ͷ���
for i=1:4
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
Is1=reshape(ICAeds(1,:),[100,100,3]);
Is2=reshape(ICAeds(2,:),[100,100,3]);
Is3=reshape(ICAeds(3,:),[100,100,3]);
Is4=reshape(ICAeds(4,:),[100,100,3]);

II1=uint8(round(Is1));
II2=uint8(round(Is2));
II3=uint8(round(Is3));
II4=uint8(round(Is4));
%��ʾ������ͼ��
figure(3)
subplot(141),imshow(II1),title( ' �� �� �� �� ͼ Ƭ1' );%,subplot(322),imhist(II1),title( ' ����ͼƬ 1��ֱ��ͼ ' );
subplot(142),imshow(II2),title( ' �� �� �� �� ͼ Ƭ2' );%,subplot(324),imhist(II2),title( ' ����ͼƬ 2��ֱ��ͼ ' );
subplot(143),imshow(II3),title( ' �� �� �� �� ͼ Ƭ3' );%,subplot(326),imhist(II3),title( ' ����ͼƬ 3��ֱ��ͼ ' );
subplot(144),imshow(II4),title( ' �� �� �� �� ͼ Ƭ3' );

III1=imsubtract(I1,II1);
III2=imsubtract(I2,II2);
III3=imsubtract(I3,II3);
III4=imsubtract(I4,II4);
%��ʾ������ͼƬ��ԭͼ�Ĳ�ֵͼ�Լ���Ӧ��ֱ��ͼ %
% figure(4)
% subplot(321),imshow(III1),title( ' �� ֵ ͼ Ƭ1' ),subplot(322);%,imhist(III1),title( ' ��ֵͼƬ 1��ֱ��ͼ ' );
% subplot(323),imshow(III2),title( ' �� ֵ ͼ Ƭ2' ),subplot(324);%,imhist(III2),title( ' ��ֵͼƬ 2��ֱ��ͼ ' );
% subplot(325),imshow(III3),title( ' �� ֵ ͼ Ƭ3' ),subplot(326);%,imhist(III3),title( ' ��ֵͼƬ 3��ֱ��ͼ ' );
