import cv2

'''
RGB to Gray
'''
im = cv2.imread(r"me.jpg")
cv2.imshow("BGR", im)
# 使用cvtColor进行颜色空间变化 cv2.COLOR_BGR2GRAY 代表BGR to gray
img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", img_gray)
cv2.waitKey()
cv2.destroyAllWindows()

'''
BGR to RGB
'''
# 使用cvtColor进行颜色空间变化 cv2.COLOR_BGR2RGB 代表BGR to RGB
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# 当图像数据为3通道时，imshow函数认为数据是BGR的
# 使用imshow显示RGB数据，会发现图片显示颜色畸变
cv2.imshow("RGB", im_rgb)
cv2.waitKey()
cv2.destroyAllWindows()

'''
BGR to HSV
'''
# 使用cvtColor进行颜色空间变化 cv2.COLOR_BGR2HSV 代表BGR to HSV
im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
# 当图像数据为3通道时，imshow函数认为数据是BGR的
# 使用imshow显示HSV数据，会将HSV分量强行当做BGR进行显示
cv2.imshow("HSV", im_hsv)
cv2.waitKey()
cv2.destroyAllWindows()
