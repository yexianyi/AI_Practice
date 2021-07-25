import cv2
from matplotlib import pyplot as plt
# 读取并显示图像
im = cv2.imread("me.jpg",0)
cv2.imshow('org', im)

# 绘制灰度图像的直方图
plt.hist(im.ravel(), 256, [0,256])
plt.show()


'''
直方图均衡化
'''
# 调用OpenCV的直方图均衡化API
im_equ1 = cv2.equalizeHist(im)
cv2.imshow('equal', im_equ1)

# 显示原始图像的直方图
plt.subplot(2,1,1)
plt.hist(im.ravel(), 256, [0,256],label='org')
plt.legend()

# 显示均衡化图像的直方图
plt.subplot(2,1,2)
plt.hist(im_equ1.ravel(), 256, [0,256],label='equalize')
plt.legend()
plt.show()

# 等待用户按键反馈后销毁窗口
cv2.waitKey()
cv2.destroyAllWindows()
