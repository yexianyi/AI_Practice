import numpy as np
import cv2

im = cv2.imread('me.jpg')
cv2.imshow("orig", im)

# 进行水平镜像
im_flip0 = cv2.flip(im, 0)
cv2.imshow("flip vertical ", im_flip0)

# 进行垂直镜像
im_flip1 = cv2.flip(im, 1)
cv2.imshow("flip horizontal ", im_flip1)

cv2.waitKey()
cv2.destroyAllWindows()
