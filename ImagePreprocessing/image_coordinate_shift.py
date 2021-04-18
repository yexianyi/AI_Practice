import numpy as np
import cv2


# 定义平移translate函数
def translate(img, x, y):
    # 获取图像尺寸
    (h, w) = img.shape[:2]

    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])

    # 使用OpenCV仿射变换函数实现平移操作
    shifted = cv2.warpAffine(img, M, (w, h))

    # 返回转换后的图像
    return shifted


# 加载图像并显示
im = cv2.imread('me.jpg')
cv2.imshow("Orig", im)

# 对原图做平移操作
# 下移50像素
shifted = translate(im, 0, 50)
cv2.imshow("Shift1", shifted)
# 左移100像素
shifted = translate(im, -100, 0)
cv2.imshow("Shift2", shifted)
# 右移50，下移100像素
shifted = translate(im, 50, 100)
cv2.imshow("Shift3", shifted)

cv2.waitKey()
cv2.destroyAllWindows()
