import cv2
from matplotlib import pyplot as plt

img = cv2.imread('me.jpg', 0);  # 打开为灰度图像
plt.imshow(img, 'gray')  # 必须规定为显示的为什么图像
# plt.xticks([]),plt.yticks([]) #隐藏坐标线
plt.show()  # 显示出来，不要也可以
