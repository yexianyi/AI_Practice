'''
中值滤波
'''
import cv2
import numpy as np

im = cv2.imread('me.jpg')
cv2.imshow('org', im)

# 调用OpenCV的中值模糊API
im_medianblur = cv2.medianBlur(im, 5)

cv2.imshow('median_blur', im_medianblur)

cv2.waitKey(0)
cv2.destroyAllWindows()
