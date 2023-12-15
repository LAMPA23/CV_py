import os
import cv2
import math
import copy

os.system('cls')

def logarithmer(image, log_base):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for RGB in range(3):
                image[y,x,RGB] = math.log(image[y,x,RGB] + 1, log_base) * 50
            

original_image = cv2.imread('pic2.jpg')
log_e_x50_image = copy.deepcopy(original_image)
log_20_x50_image = copy.deepcopy(original_image)

logarithmer(log_e_x50_image, 2.72)
logarithmer(log_20_x50_image, 20)

cv2.imshow('original_image', original_image)
cv2.imshow('log_e_x50_image', log_e_x50_image)
cv2.imshow('log_20_x50_image', log_20_x50_image)
cv2.waitKey(0)
