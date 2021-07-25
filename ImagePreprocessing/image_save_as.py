import cv2

image = cv2.imread('me.jpg')  # 将‘flower.jpg’的图片与.py文件放在同一目录下，或者使用绝对路径
cv2.imwrite('me2.png', image)
