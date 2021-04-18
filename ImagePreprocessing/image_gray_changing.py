'''
灰度变化。反转，灰度拉伸，灰度压缩
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt


# 定义线性灰度变化函数
# k>1时 实现灰度数值的拉伸
# 0<k<1时 实现灰度数值的压缩
# k=-1 b=255 实现灰度反转
def linear_trans(img, k, b=0):
    # 计算灰度线性变化的映射表
    trans_list = [(np.float32(x) * k + b) for x in range(256)]
    # 将列表转换为np.array
    trans_table = np.array(trans_list)
    # 将超过[0,255]灰度范围的数值进行调整,并指定数据类型为uint8
    trans_table[trans_table > 255] = 255
    trans_table[trans_table < 0] = 0
    trans_table = np.round(trans_table).astype(np.uint8)
    # 使用OpenCV的look up table函数修改图像的灰度值
    return cv2.LUT(img, trans_table)


im = cv2.imread('me.jpg', 0)
cv2.imshow('org', im)

# 反转
im_inversion = linear_trans(im, -1, 255)
cv2.imshow('inversion', im_inversion)
# 灰度拉伸
im_stretch = linear_trans(im, 1.2)
cv2.imshow('graystretch', im_stretch)
# 灰度压缩
im_compress = linear_trans(im, 0.8)
cv2.imshow('graycompress', im_compress)
cv2.waitKey()
cv2.destroyAllWindows()
