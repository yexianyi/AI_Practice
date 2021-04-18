import cv2

# 读取一副图像 第一个参数是图像路径
# 第二个参数代表读取方式，1表示3通道彩色，0表示单通道灰度
im = cv2.imread(r"me.jpg", 1)
# 在"test"窗口中显示图像im
cv2.imshow("test", im)
# 等待用户按键反馈
cv2.waitKey()
# 销毁所有创建的窗口
cv2.destroyAllWindows()
# 打印图像数据的数据结构类型
print(type(im))
# 打印图像的尺寸
print(im.shape)
# 将图像保存到指定路径
# cv2.imwrite('lena.jpg',im)
