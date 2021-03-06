import cv2
import numpy as np

im = cv2.imread('me.jpg')
cv2.imshow('org', im)
# 锐化算子
sharpen_1 = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]])
# 使用filter2D进行滤波操作
im_sharpen1 = cv2.filter2D(im, -1, sharpen_1)
cv2.imshow('sharpen_1', im_sharpen1)

# 锐化算子2
sharpen_2 = np.array([
    [0, -1, 0],
    [-1, 8, -1],
    [0, 1, 0]]) / 4.0

# 使用filter2D进行滤波操作
im_sharpen2 = cv2.filter2D(im, -1, sharpen_2)
cv2.imshow('sharpen_2', im_sharpen2)

cv2.waitKey(0)
cv2.destroyAllWindows()
