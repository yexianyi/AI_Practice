import cv2
import numpy as np

garyImage = cv2.imread('me.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('me2.png', garyImage)
