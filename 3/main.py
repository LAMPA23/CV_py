import colorgrad 
import skimage.color  as clr

import os
import cv2

import image_show as im_show
import image_converter as im_conv



os.system('cls')
for dir in os.listdir():
    if dir != 'original_images':
        os.system(f'rd /s /q {dir}')


image_1_path = 'original_images/pic1.png'
image_2_path = 'original_images/pic2.png'
original_image_1 = cv2.imread(image_1_path)
original_image_2 = cv2.imread(image_2_path)


im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_BGR2RGB), 'BGR to RGB', 'conv')
im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_BGR2XYZ), 'BGR to XYZ', 'conv')
im_show.save_image(im_conv.BGR_to_CMY(original_image_1), 'BGR to CMY', 'conv')

im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_BGR2HSV), 'BGR to HSV', 'conv')
im_show.save_image(im_conv.BGR_to_HSI(original_image_1), 'BGR to HSI', 'conv')

im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_BGR2Lab), 'BGR to Lab', 'conv')
im_show.save_image(cv2.cvtColor(original_image_1, cv2.COLOR_BGR2Luv), 'BGR to Luv', 'conv')
