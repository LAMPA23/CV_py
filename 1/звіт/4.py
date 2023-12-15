import cv2
import copy
import math
import os
import numpy as np

os.system('cls')

def power_law_transformation(image, max_val_for_pixel):
    for RGB in range(3):
        min_val = np.min(image[:,:,RGB])
        max_val = np.max(image[:,:,RGB])
        image[:,:,RGB] = ((image[:,:,RGB] - min_val) / (max_val - min_val)) * max_val_for_pixel
    
original_image = cv2.imread('pic4.jpg')
image_1 = copy.deepcopy(original_image)
image_2 = copy.deepcopy(original_image)

# Застосувати степеневе перетворення
power_law_transformation(image_1, 255)
power_law_transformation(image_2, 200)

# Відобразити оригінальне та перетворене зображення
cv2.imshow('Original Image', original_image)
cv2.imshow('Image 1', image_1)
cv2.imshow('Image 2', image_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
