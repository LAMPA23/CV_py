import os 
import cv2
import numpy as np
import my_task_functions as mtf
import my_general_purpose_functions as mgpf
import skimage.morphology as morphology
import skimage.measure as measure

os.system('cls')

kernal_path = 'C:/Users/bimee/OneDrive/computer vision/5/original_images/'

original_image_1 = mgpf.get_gray_image(kernal_path, 'pic.1.tif')
original_image_2 = mgpf.get_gray_image(kernal_path, 'pic.2.tif')
original_image_3 = mgpf.get_gray_image(kernal_path, 'pic.3.tif')
original_image_4 = mgpf.get_gray_image(kernal_path, 'pic.4.tif')
original_image_5 = mgpf.get_gray_image(kernal_path, 'pic.5.tif')
original_image_6 = mgpf.get_gray_image(kernal_path, 'pic.6.tif')
original_image_7 = mgpf.get_gray_image(kernal_path, 'pic.7.tif')
original_image_8_png = mgpf.get_gray_image(kernal_path, 'pic.8.png')
original_image_8_jpg = mgpf.get_gray_image(kernal_path, 'pic.8.jpg')


# Task #1 Виявити точки на зображенні (файл – pic.1.tif)
mgpf.show_images_x2(
    original_image_1,
    mtf.detect_points(original_image_1),
    'Task #1 Виявити точки на зображенні (файл – pic.1.tif)',
    'Оригінальне зображення',
    'Оброблене зображення')


