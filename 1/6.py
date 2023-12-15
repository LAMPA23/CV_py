import cv2
import numpy as np
import os

os.system('cls')

def smoothed_image(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)

image = cv2.imread('pic6.jpg')

cv2.imshow('Original Image', image)
cv2.imshow('Smoothed 5x5 Image', smoothed_image(image, 5))
cv2.imshow('Smoothed 10x10 Image', smoothed_image(image, 10))
cv2.waitKey(0)
cv2.destroyAllWindows()
