import cv2
import copy
import math
import numpy as np

def power_law_transformation(image, gamma, c=1):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for RGB in range(3):
                image[y, x, RGB] = min(c * math.pow(image[y, x, RGB], gamma), 255)
                
original_image = cv2.imread('pic3.jpg')
image_1 = copy.deepcopy(original_image)
image_2 = copy.deepcopy(original_image)

# Визначити параметри степеневого перетворення
gamma_1 = 0.5  # Параметр гамма
c_1 = 2  # Константа масштабування
gamma_2 = 2  # Параметр гамма
c_2 = 2  # Константа масштабування


# Застосувати степеневе перетворення
power_law_transformation(image_1, gamma_1, c_1)
power_law_transformation(image_2, gamma_2, c_2)

# Відобразити оригінальне та перетворене зображення
cv2.imshow('Original Image', original_image)
cv2.imshow('Image 1', image_1)
cv2.imshow('Image 2', image_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
