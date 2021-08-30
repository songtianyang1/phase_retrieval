# -*- codeing = utf-8 -*-
# @Time : 2020/10/20 0020 16:57
# @Author : 天洋
# @File : hunhe.py
# @Software : PyCharm


import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import cv2
from phase_retrieval_GS import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import misc

img1 = cv2.imread('x.jpg')
img2 = cv2.imread('v.jpg')
img3 = cv2.imread('z.jpg')
img4 = cv2.imread('w.jpg')
print(img1.shape)
print(img2.shape)
print(img3.shape)
print(img4.shape)
#
x1=np.reshape(img1,(-1,1))  #列向量
x2=np.reshape(img2,(-1,1))
x3=np.reshape(img3,(-1,1))  #列向量
x4=np.reshape(img4,(-1,1))
print(len(x1))
print(len(x2))
print(x1.shape)
print(x2.shape)

A=np.array([[2,3,4,5],[4,5,6,7],[6,7,8,9],[8,9,2,3]])
B=np.hstack((x1,x2,x3,x4))
print(B.shape)
C=B.transpose()
print(A.shape)
print(C.shape)
# a=A*C   #混合信号
a=np.dot(A,C)
a=a.transpose()
a=np.array(a)
a1=a[:,0]
a1=np.reshape(a1,(100, 100, 3))
a1 = (a1-np.min(a1)) / (np.max(a1) - np.min(a1))
a2=a[:,1]
a2=np.reshape(a2,(100, 100, 3))
a2 = (a2-np.min(a2)) / (np.max(a2) - np.min(a2))
a3=a[:,2]
a3=np.reshape(a3,(100, 100, 3))
a3 = (a3-np.min(a3)) / (np.max(a3) - np.min(a3))
a4=a[:,3]
a4=np.reshape(a4,(100, 100, 3))
a4 = (a4-np.min(a4)) / (np.max(a4) - np.min(a4))


print(a1.shape)
print(a2.shape)



#a1去相位

aa=[a1,a2,a3,a4]

b, g, r = cv2.split(a1)
lista1 = [b, g, r]
lista11 = []
lista12 = []
max_iters = 1000
for i in range(len(lista1)):
    print(len(lista1))
    p = Ger_Sax_algo(lista1[i], max_iters)
    lista11.append(p)  # list1=[b, g, r]



for a in range(len(lista11)):
    # print(len(list1))
    recovery = np.fft.ifft2(np.exp(lista11[a] * 1j))

    q = (np.absolute(recovery) ** 2)
    lista12.append(q)  # list2=[b,g,r]
# print(len(list2))

merged_a1 = cv2.merge([lista12[2], lista12[1], lista12[0]])
merged_normed_a1 = (merged_a1-np.min(merged_a1)) / (np.max(merged_a1) - np.min(merged_a1))

#a2去相位
b, g, r = cv2.split(a2)
lista2 = [b, g, r]
lista21 = []
lista22 = []
max_iters = 1000
for i in range(len(lista2)):
    print(len(lista2))
    p = Ger_Sax_algo(lista2[i], max_iters)
    lista21.append(p)  # list1=[b, g, r]



for a in range(len(lista21)):
    # print(len(list1))
    recovery = np.fft.ifft2(np.exp(lista21[a] * 1j))

    q = (np.absolute(recovery) ** 2)
    lista22.append(q)  # list2=[b,g,r]
# print(len(list2))

merged_a2 = cv2.merge([lista22[2], lista22[1], lista22[0]])
merged_normed_a2 = (merged_a2-np.min(merged_a2)) / (np.max(merged_a2) - np.min(merged_a2))

# a3去相位
b, g, r = cv2.split(a3)
lista3 = [b, g, r]
lista31 = []
lista32 = []
max_iters = 1000
for i in range(len(lista3)):
    print(len(lista3))
    p = Ger_Sax_algo(lista3[i], max_iters)
    lista31.append(p)  # list1=[b, g, r]



for a in range(len(lista31)):
    # print(len(list1))
    recovery = np.fft.ifft2(np.exp(lista31[a] * 1j))

    q = (np.absolute(recovery) ** 2)
    lista32.append(q)  # list2=[b,g,r]
# print(len(list2))

merged_a3 = cv2.merge([lista32[2], lista32[1], lista32[0]])
merged_normed_a3 = (merged_a3-np.min(merged_a3)) / (np.max(merged_a3) - np.min(merged_a3))


# a4去相位
b, g, r = cv2.split(a4)
lista4 = [b, g, r]
lista41 = []
lista42 = []
max_iters = 1000
for i in range(len(lista4)):
    print(len(lista4))
    p = Ger_Sax_algo(lista4[i], max_iters)
    lista41.append(p)  # list1=[b, g, r]



for a in range(len(lista41)):
    # print(len(list1))
    recovery = np.fft.ifft2(np.exp(lista41[a] * 1j))

    q = (np.absolute(recovery) ** 2)
    lista42.append(q)  # list2=[b,g,r]
# print(len(list2))

merged_a4 = cv2.merge([lista42[2], lista42[1], lista42[0]])
merged_normed_a4 = (merged_a4-np.min(merged_a4)) / (np.max(merged_a4) - np.min(merged_a4))


"原图及分离信号"
plt.figure(1)

plt.subplot(241)
img1 = img1[:, :, [2, 1, 0]]
plt.imshow(img1)
plt.axis('off')
plt.title('a')

plt.subplot(242)
img2 = img2[:, :, [2, 1, 0]]
plt.imshow(img2)
plt.axis('off')
plt.title('b')

plt.subplot(243)
img3 = img3[:, :, [2, 1, 0]]
plt.imshow(img3)
plt.axis('off')
plt.title('c')

plt.subplot(244)
img4 = img4[:, :, [2, 1, 0]]
plt.imshow(img4)
plt.axis('off')
plt.title('d')

plt.subplot(245)
a1 = a1[:, :, [2, 1, 0]]
plt.imshow(a1)
plt.axis('off')
plt.title('a1')

plt.subplot(246)
a2 = a2[:, :, [2, 1, 0]]
plt.imshow(a2)
plt.axis('off')
plt.title('a2')

plt.subplot(247)
a3 = a3[:, :, [2, 1, 0]]
plt.imshow(a3)
plt.axis('off')
plt.title('a3')

plt.subplot(248)
a4 = a4[:, :, [2, 1, 0]]
plt.imshow(a4)
plt.axis('off')
plt.title('a4')





"掩码后分离图"
plt.figure(2)

plt.subplot(241)
# merged_normed_1 = merged_normed_1[:, :, [2, 1, 0]]

plt.imshow(a1)
plt.title('a1')

plt.subplot(242)
plt.imshow(lista11[0])
plt.title('r1')

plt.subplot(243)
plt.imshow(lista11[1])
plt.title('g1')

plt.subplot(244)
plt.imshow(lista11[2])
plt.title('b1')

plt.subplot(245)
# merged_normed_1 = merged_normed_1[:, :, [2, 1, 0]]

plt.imshow(a2)
plt.title('a2')

plt.subplot(246)
plt.imshow(lista21[0])
plt.title('r1')

plt.subplot(247)
plt.imshow(lista21[1])
plt.title('g1')

plt.subplot(248)
plt.imshow(lista21[2])
plt.title('b1')

plt.figure(3)

plt.subplot(241)
# merged_normed_1 = merged_normed_1[:, :, [2, 1, 0]]

plt.imshow(a3)
plt.title('a3')

plt.subplot(242)
plt.imshow(lista31[0])
plt.title('r1')

plt.subplot(243)
plt.imshow(lista31[1])
plt.title('g1')

plt.subplot(244)
plt.imshow(lista31[2])
plt.title('b1')

plt.subplot(245)
# merged_normed_1 = merged_normed_1[:, :, [2, 1, 0]]

plt.imshow(a4)
plt.title('a4')

plt.subplot(246)
plt.imshow(lista41[0])
plt.title('r1')

plt.subplot(247)
plt.imshow(lista41[1])
plt.title('g1')

plt.subplot(248)
plt.imshow(lista41[2])
plt.title('b1')



"恢复后分离图"
plt.figure(4)

plt.subplot(241)
plt.imshow(merged_normed_a1)
plt.title('Recovered image_a1')

plt.subplot(242)
plt.imshow(lista12[0], cmap="gray")
plt.axis('off')
plt.title('b2')


plt.subplot(243)
plt.imshow(lista12[1], cmap="gray")
plt.axis('off')
plt.title('g2')

plt.subplot(244)
plt.imshow(lista12[2], cmap="gray")
plt.axis('off')
plt.title('r2')

plt.subplot(245)
plt.imshow(merged_normed_a2)
plt.title('Recovered image_a2')

plt.subplot(246)

plt.imshow(lista22[0], cmap="gray")
plt.axis('off')
plt.title('b2')


plt.subplot(247)
plt.imshow(lista22[1], cmap="gray")
plt.axis('off')
plt.title('g2')

plt.subplot(248)
plt.imshow(lista22[2], cmap="gray")
plt.axis('off')
plt.title('r2')

plt.figure(5)

plt.subplot(241)
plt.imshow(merged_normed_a3)
plt.title('Recovered image_a3')

plt.subplot(242)
plt.imshow(lista32[0], cmap="gray")
plt.axis('off')
plt.title('b2')


plt.subplot(243)
plt.imshow(lista32[1], cmap="gray")
plt.axis('off')
plt.title('g2')

plt.subplot(244)
plt.imshow(lista32[2], cmap="gray")
plt.axis('off')
plt.title('r2')

plt.subplot(245)
plt.imshow(merged_normed_a4)
plt.title('Recovered image_a4')

plt.subplot(246)

plt.imshow(lista42[0], cmap="gray")
plt.axis('off')
plt.title('b2')


plt.subplot(247)
plt.imshow(lista42[1], cmap="gray")
plt.axis('off')
plt.title('g2')

plt.subplot(248)
plt.imshow(lista42[2], cmap="gray")
plt.axis('off')
plt.title('r2')


#
#
# plt.figure(6)
#
# plt.subplot()
# plt.imshow(lista12[2], cmap="gray")
# plt.axis('off')
# plt.savefig('a1r.jpg')
# plt.title('a1r')
# plt.figure(7)
# plt.subplot()
# # merged_normed_1 = merged_normed_1[:, :, [2, 1, 0]]
# plt.imshow(lista22[2], cmap="gray")
# plt.axis('off')
# plt.savefig('a2r.jpg')
# plt.title('a2r')
#
# plt.figure(8)
#
# plt.subplot()
# plt.imshow(lista32[2], cmap="gray")
# plt.axis('off')
# plt.savefig('a3r.jpg')
# plt.title('a3r')
# plt.figure(9)
# plt.subplot()
# # merged_normed_1 = merged_normed_1[:, :, [2, 1, 0]]
# plt.imshow(lista42[2], cmap="gray")
# plt.axis('off')
# plt.savefig('a4r.jpg')
# plt.title('a4r')

plt.figure(6)

plt.subplot()
plt.imshow(merged_normed_a1)
# plt.title('Recovered image_a1')
plt.axis('off')
# 去白边  保存
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig('C:/Users/Administrator/Documents/MATLAB/fastica/hunhe04_a1.jpg',bbox_inches='tight',transparent=True, dpi=300, pad_inches = 0)

plt.figure(7)

plt.subplot()
plt.imshow(merged_normed_a2)
# plt.title('Recovered image_a2')
plt.axis('off')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig('C:/Users/Administrator/Documents/MATLAB/fastica/hunhe04_a2.jpg',bbox_inches='tight',transparent=True, dpi=300, pad_inches = 0)

plt.figure(8)

plt.subplot()
plt.imshow(merged_normed_a3)
# plt.title('Recovered image_a1')
plt.axis('off')
# 去白边  保存
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig('C:/Users/Administrator/Documents/MATLAB/fastica/hunhe04_a3.jpg',bbox_inches='tight',transparent=True, dpi=300, pad_inches = 0)

plt.figure(9)

plt.subplot()
plt.imshow(merged_normed_a4)
# plt.title('Recovered image_a2')
plt.axis('off')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig('C:/Users/Administrator/Documents/MATLAB/fastica/hunhe04_a4.jpg',bbox_inches='tight',transparent=True, dpi=300, pad_inches = 0)



plt.show()
