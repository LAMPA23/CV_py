import os
import cv2

increase_value = 30 

os.system('cls')

original_image = cv2.imread('pic1.jpg')

negative_image = ~original_image

negative_and_increasing_image = negative_image
for y in range(negative_and_increasing_image.shape[0]):
    for x in range(negative_and_increasing_image.shape[1]):
        if(negative_and_increasing_image[y,x][0] > 255 - increase_value):
            negative_and_increasing_image[y,x] = 255 - increase_value   
negative_and_increasing_image = negative_and_increasing_image + increase_value


cv2.imshow('original_image', original_image)
cv2.imshow('negative_image', negative_image)
cv2.imshow('negative_and_increasing_image', negative_and_increasing_image)

cv2.waitKey(0)