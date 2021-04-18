import numpy as np
import cv2

im = cv2.imread('me.jpg')
cv2.imshow("orig", im)

# 获取图像尺寸
(h, w) = im.shape[:2]

# 缩放的目标尺寸
dst_size = (200, 300)

# 最邻近插值
method = cv2.INTER_NEAREST

# 进行缩放
resized = cv2.resize(im, dst_size, interpolation=method)
cv2.imshow("resized1", resized)

# 缩放的目标尺寸
dst_size = (800, 600)

# 双线性插值
method = cv2.INTER_LINEAR

# 进行缩放
resized = cv2.resize(im, dst_size, interpolation=method)
cv2.imshow("resized2", resized)

cv2.waitKey()
cv2.destroyAllWindows()
