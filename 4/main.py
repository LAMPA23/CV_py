import os 
import cv2
import my_functions as mf
import skimage.morphology as morphology
import skimage.measure as measure

os.system('cls')

kernal_path = 'C:/Users/bimee/OneDrive/computer vision/4/original_images/'

original_image_1 = mf.get_gray_image(kernal_path, 'pic.1.jpg')
original_image_2 = mf.get_gray_image(kernal_path, 'pic.2.jpg')
original_image_3 = mf.get_gray_image(kernal_path, 'pic.3.jpg')
original_image_4 = mf.get_gray_image(kernal_path, 'pic.4.jpg')
original_image_5 = mf.get_gray_image(kernal_path, 'pic.5.jpg')
original_image_6a = mf.get_gray_image(kernal_path, 'pic.6a.tif')
original_image_6b = mf.get_gray_image(kernal_path, 'pic.6b.tif')
original_image_7 = mf.get_gray_image(kernal_path, 'pic.7.png')
original_image_8 = mf.get_gray_image(kernal_path, 'pic.8.png')
original_image_9 = mf.get_gray_image(kernal_path, 'pic.9.png')
original_image_10 = mf.get_gray_image(kernal_path, 'pic.10.png')
original_image_rice = mf.get_gray_image(kernal_path, 'rice.10.png')

# Tast #1

