import colorgrad 
import skimage.color  as clr

import os
import cv2

import image_show as im_show
import image_converter as im_conv


os.system('cls')
for dir in os.listdir():
    if dir != 'original_images':
        os.system(f'rd /s /q "{dir}"')
os.system('cls')


image_1_path = 'original_images/pic1.png'
image_2_path = 'original_images/pic2.png'
original_image_1 = cv2.imread(image_1_path)
original_image_2 = cv2.imread(image_2_path)
print(f'pic1.png type mode is {im_conv.get_color_space(image_1_path)}')
print(f'pic2.png type mode is {im_conv.get_color_space(image_2_path)}')


# Task #1
im_show.save_image(original_image_1, 'Original (RGB)', 'Task #1')
im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_RGB2BGR), 'RGB to BGR', 'Task #1')
im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_RGB2XYZ), 'RGB to XYZ', 'Task #1')
im_show.save_image(im_conv.BGR_to_CMY(original_image_1), 'RGB to CMY', 'Task #1')
im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_RGB2HSV), 'RGB to HSV', 'Task #1')
im_show.save_image(im_conv.BGR_to_HSI(original_image_1), 'RGB to HSI', 'Task #1')
im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_RGB2Lab), 'RGB to Lab', 'Task #1')
im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_RGB2Luv), 'RGB to Luv', 'Task #1')


# Task #2
im_show.save_image(original_image_2, 'Original', 'Task #2')
im_show.save_image(im_conv.to_smooth(original_image_2), 'Smooth', 'Task #2')

