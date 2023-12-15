import os
import cv2
import numpy as np

os.system('cls')

def gradient_processing(image, derivative_order):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, derivative_order, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, derivative_order, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return cv2.convertScaleAbs(gradient_magnitude)
 
image = cv2.imread('pic8.jpg')

y_max, x_max, _ = image.shape
scale = 2
new_size = (int(y_max / scale), int(x_max / scale))

cv2.imshow('Original Image', cv2.resize(image, new_size))
cv2.imshow('Image 1', cv2.resize(gradient_processing(image, 1), new_size))
cv2.imshow('Image 2', cv2.resize(gradient_processing(image, 2), new_size))
cv2.waitKey(0)
cv2.destroyAllWindows()