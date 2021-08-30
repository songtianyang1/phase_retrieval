import math
import numpy as np


def Ger_Sax_algo(im, max_iter):
    h, w = im.shape  # 查看矩阵或者数组的维数
    pm_s = np.random.rand(h, w)  # 返回一个或一组服从“0~1”均匀分布的随机样本，h行w列矩阵
    pm_f = np.ones((h, w))  # 创建h行w列矩阵
    am_s = np.sqrt(im)  # 每个元素的开方
    am_f = np.ones((h, w))  # 创建h行w列矩阵

    signal_s = am_s * np.exp(pm_s * 1j)  # e的幂次方

    for iter in range(max_iter):
        signal_f = np.fft.fft2(signal_s)  # 计算二维的傅里叶变换
        pm_f = np.angle(signal_f)  # 计算复数的辐角主值
        signal_f = am_f * np.exp(pm_f * 1j)  # e的幂次方
        signal_s = np.fft.ifft2(signal_f)  # 傅里叶逆变换
        pm_s = np.angle(signal_s)  # 计算复数的辐角主值
        signal_s = am_s * np.exp(pm_s * 1j)

    pm = pm_f
    return pm
