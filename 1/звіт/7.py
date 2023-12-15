import cv2
import numpy as np
import os

os.system('cls')

def apply_laplacian_manually(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    y_max, x_max = image.shape
    laplacian_result = np.zeros_like(image, dtype=np.float32)
    for y in range(1, y_max - 1):
        for x in range(1, x_max - 1):
            laplacian_result[y, x] = np.sum(image[y-1:y+2, x-1:x+2] * mask)
    return cv2.convertScaleAbs(laplacian_result)
  

image = cv2.imread('pic7.jpg')

laplacian_mask = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
mdf_mask = np.array([[0, -2, 0], [-2, 9, -2], [0, -2, 0]])

laplacian_mask_image = apply_laplacian_manually(image, laplacian_mask)
mdf_mask_image = apply_laplacian_manually(image, mdf_mask)

cv2.imshow('Original Image', image)
cv2.imshow('Image_1', laplacian_mask_image)
cv2.imshow('Image_2', mdf_mask_image)
cv2.waitKey(0)
cv2.destroyAllWindows()