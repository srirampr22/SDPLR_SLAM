import cv2
import numpy as np

# img = cv2.imread('sampledand.png', 0)
img = cv2.imread('sampledand.png')

kernel = np.ones((5,5), np.uint8)

# erosion first, then dilation
# img_erosion = cv2.erode(img, kernel, iterations=1)
# # img_dilation = cv2.dilate(img, kernel, iterations=1)
# img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

# dilation first, then erosion
img_dilation = cv2.dilate(img, kernel, iterations=1)
img_erosion = cv2.erode(img_dilation, kernel, iterations=1)

cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)

cv2.waitKey(0)