import cv2
import numpy as np

img = np.zeros((3, 3), dtype=np.uint8)  # 通过二维NumPy数组来简单创建一个黑色的正方形图像
print(img)  # 在控制台打印该图像
print(img.shape)  # 通过shape属性来查看图像的结构，返回行和列，如果有一个以上的通道，还会返回通道数

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 利用cv2.cvtColor函数将该图像转换成BGR格式
print(img)
print(img.shape)
cv2.namedWindow("Image")  # 显示该图像
cv2.imshow("Image", img)
cv2.waitKey(0)
