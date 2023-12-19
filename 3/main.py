import colorgrad 
import skimage.color  as clr

import os
import cv2

import image_show as im_show
import image_converter as im_conv


# Folder correcting
os.system('cls')
for dir in os.listdir():
    if dir != 'original_images':
        os.system(f'rd /s /q "{dir}"')
os.system('cls')


# Get origin image
image_1_path = 'original_images/pic1.png'
image_2_path = 'original_images/pic2.png'
original_image_1 = cv2.imread(image_1_path)
original_image_2 = cv2.imread(image_2_path)
print(f'pic1.png color space is {im_conv.get_color_space(image_1_path)}')
print(f'pic2.png color space is {im_conv.get_color_space(image_2_path)}')


# Task #1
im_show.save_image(original_image_1, 'Original (RGB)', 'Task #1')
im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_RGB2BGR), 'RGB to BGR', 'Task #1')
im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_RGB2XYZ), 'RGB to XYZ', 'Task #1')
im_show.save_image(im_conv.RGB_to_CMY(original_image_1), 'RGB to CMY', 'Task #1')
im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_RGB2HSV), 'RGB to HSV', 'Task #1')
im_show.save_image(im_conv.RGB_to_HSI(original_image_1), 'RGB to HSI', 'Task #1')
im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_RGB2Lab), 'RGB to Lab', 'Task #1')
im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_RGB2Luv), 'RGB to Luv', 'Task #1')


# Task #2
im_show.save_image(original_image_2, 'Original', 'Task #2')
im_show.save_image(im_conv.smooth_RGB(original_image_2), 'Smooth', 'Task #2')


# Task #3
im_show.save_image(original_image_2, 'Original', 'Task #3')
im_show.save_image(im_conv.smooth_HSV(original_image_2), 'Smooth HSV', 'Task #3')
im_show.save_image(im_conv.smooth_Lab(original_image_2), 'Smooth Lab', 'Task #3')


# Task #4
im_show.save_image(original_image_2, 'Original', 'Task #4')
im_show.save_image(im_conv.increase_sharpness_RGB(original_image_2), 'increase sharpness RGB', 'Task #4')


# Task #5
im_show.save_image(original_image_2, 'Original', 'Task #5')
im_show.save_image(im_conv.increase_sharpness_HSV(original_image_2), 'increase_sharpness_HSV', 'Task #5')
im_show.save_image(im_conv.increase_sharpness_Lab(original_image_2), 'increase_sharpness_Lab', 'Task #5')


# Task #6
im_show.save_image(original_image_2, 'Original', 'Task #6')
im_show.save_image(im_conv.equaliz_RGB(original_image_2), 'Equalized RGB', 'Task #6')
im_show.save_image(im_conv.equaliz_HSV(original_image_2), 'Equalized HSV', 'Task #6')
im_show.save_image(im_conv.equaliz_Lab(original_image_2), 'Equalized Lab', 'Task #6')


# Task #7
im_show.save_image(original_image_2, 'Original', 'Task #7')
im_show.save_image(im_conv.get_per_plane_gradient(original_image_2), 'get_per_plane_gradient', 'Task #7')
im_show.save_image(im_conv.get_sobel_gradient_normalized(original_image_2), 'get_sobel_gradient_normalized', 'Task #7')
im_show.save_image(im_conv.get_absolute_difference(original_image_2), 'get_absolute_difference', 'Task #7')

