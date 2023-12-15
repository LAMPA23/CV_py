import cv2
import numpy as np
import os

os.system('cls')

def manual_median_blur(image, kernel_size):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result = np.zeros_like(image)
    padding = kernel_size // 2
    for y in range(padding, image.shape[0] - padding):
        for x in range(padding, image.shape[1] - padding):
            neighborhood = image[y - padding:y + padding + 1, x - padding:x + padding + 1]
            result[y, x] = np.median(neighborhood[:, :])
    return result

image = cv2.imread('pic9.jpg')

cv2.imshow('Original Image', image)
cv2.imshow('Image 1', manual_median_blur(image, 2))
cv2.imshow('Image 2', manual_median_blur(image, 5))
cv2.waitKey(0)
cv2.destroyAllWindows()
