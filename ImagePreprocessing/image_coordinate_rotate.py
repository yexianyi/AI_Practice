import numpy as np
import cv2


# 定义旋转rotate函数
def rotate(img, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = img.shape[:2]

    # 旋转中心的缺失值为图像中心
    if center is None:
        center = (w / 2, h / 2)

    # 调用计算旋转矩阵函数
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 使用OpenCV仿射变换函数实现旋转操作
    rotated = cv2.warpAffine(img, M, (w, h))

    # 返回旋转后的图像
    return rotated


im = cv2.imread('me.jpg')
cv2.imshow("Orig", im)

# 对原图做旋转操作
# 逆时针45度
rotated = rotate(im, 45)
cv2.imshow("Rotate1", rotated)
# 顺时针20度
rotated = rotate(im, -20)
cv2.imshow("Rotate2", rotated)
# 逆时针90度
rotated = rotate(im, 90)
cv2.imshow("Rotate3", rotated)

cv2.waitKey()
cv2.destroyAllWindows()
