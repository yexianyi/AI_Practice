'''
伽马变化
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt


# 定义伽马变化函数
def gamma_trans(img, gamma):
    # 先归一化到1，做伽马计算，再还原到[0,255]
    gamma_list = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    # 将列表转换为np.array，并指定数据类型为uint8
    gamma_table = np.round(np.array(gamma_list)).astype(np.uint8)
    # 使用OpenCV的look up table函数修改图像的灰度值
    return cv2.LUT(img, gamma_table)


im = cv2.imread('me.jpg', 0)
cv2.imshow('org', im)

# 使用伽马值为0.5的变化，实现对暗部的拉升，亮部的压缩
im_gama05 = gamma_trans(im, 0.5)
cv2.imshow('gama0.5', im_gama05)
# 使用伽马值为2的变化，实现对亮部的拉升，暗部的压缩
im_gama2 = gamma_trans(im, 2)
cv2.imshow('gama2', im_gama2)
cv2.waitKey()
cv2.destroyAllWindows()
