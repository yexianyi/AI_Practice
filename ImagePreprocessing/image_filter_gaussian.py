# 方法一：直接调用OpenCV的API
import cv2
import numpy as np

im = cv2.imread('me.jpg')
cv2.imshow('org', im)

# 调用OpenCV的高斯模糊API
im_gaussianblur1 = cv2.GaussianBlur(im, (5, 5), 0)

cv2.imshow('gaussian_blur_1', im_gaussianblur1)

# 方法二：使用高斯算子和filter2D 自定义滤波操作
cv2.imshow('org', im)
# 高斯算子
gaussian_blur = np.array([
    [1, 4, 7, 4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]], np.float32) / 273

# 使用filter2D进行滤波操作
im_gaussianblur2 = cv2.filter2D(im, -1, gaussian_blur)
cv2.imshow('gaussian_blur_2', im_gaussianblur2)

cv2.waitKey(0)
cv2.destroyAllWindows()
