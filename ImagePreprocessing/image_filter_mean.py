import cv2
import numpy as np

'''
# 方法一：直接调用OpenCV的API
'''
im = cv2.imread('me.jpg')
cv2.imshow('org', im)

# 调用OpenCV的均值模糊API
im_meanblur1 = cv2.blur(im, (3, 3))
cv2.imshow('mean_blur_1', im_meanblur1)


'''
# 方法二：使用均值算子和filter2D 自定义滤波操作
'''

im = cv2.imread('me.jpg')
cv2.imshow('org', im)
# 均值算子
mean_blur = np.ones([3, 3], np.float32) / 9

# 使用filter2D进行滤波操作
im_meanblur2 = cv2.filter2D(im, -1, mean_blur)
cv2.imshow('mean_blur_2', im_meanblur2)

cv2.waitKey(0)
cv2.destroyAllWindows()
