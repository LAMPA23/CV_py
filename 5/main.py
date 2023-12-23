import os 
import cv2
import numpy as np
import my_task_functions as mtf
import my_general_purpose_functions as mgpf
from skimage import segmentation, measure, feature
import graythresh
from scipy.ndimage import distance_transform_edt


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


win_name_task_1 = 'Виявити точки на зображенні (файл – pic.1.tif)'
win_name_task_2 = 'Виявити ліній на зображенні (файл – pic.2.tif)'
win_name_task_3 = 'Виявити перепади яскравості на зображенні (файл – pic.3.tif)'
win_name_task_4 = 'Обробка з глобальним порогом (файл – pic.4.tif)'
win_name_task_5 = 'Сегментація по вододілам за допомогою перетворення відстані (файл – pic.5.tif)'
win_name_task_6 = 'Сегментація по вододілам за допомогою градієнтів (файл – pic.6.tif)'
win_name_task_7 = 'Виконати сегментацію кольорового зображення за допомогою кластеризації по k-середніх (файл – pic.8.jpg)'


# # Task #1
# images = [None] * 2
# images[0] = original_image_1
# images[1] = mtf.detect_points(original_image_1)
# mgpf.show_images([1,2], images, win_name_task_1, ['Оригінальне зображення', 'Оброблене зображення'])


# # Task #2 
# images = [None] * 6
# images[1:6] = mtf.line_finder(original_image_2)
# images[0] = original_image_2
# mgpf.show_images([2,3], images, win_name_task_2, ['Оригінальне зображення', 
#                                                   'Вертикальна обробка', 
#                                                   'Горизонтальна обробка', 
#                                                   'Права діагональ обробка', 
#                                                   'Ліва діагональ обробка', 
#                                                   'Обробка всіх напрямків']
# )


# # Task #3
# images = [None] * 2
# images[0] = original_image_3
# images[1] = feature.canny(original_image_3)
# mgpf.show_images([1,2], images, win_name_task_3, ['Оригінальне зображення','Оброблене зображення',])


# # Task #4
# images = [None] * 2
# images[0] = original_image_4
# images[1] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(original_image_4)
# mgpf.show_images([1,2], images, win_name_task_4, ['Оригінальне зображення','Оброблене зображення',])


# # Task #5
# images = [None] * 2
# images[0] = original_image_5
# images[1] = mtf.watershed_with_distance(original_image_5)
# mgpf.show_images([1,2], images, win_name_task_5, ['Оригінальне зображення', 'Оброблене зображення'])


# # Task #6
# images = [None] * 2
# images[0] = original_image_6
# images[1] = mtf.watershed_with_grad(original_image_6)
# mgpf.show_images([1,2], images, win_name_task_6, ['Оригінальне зображення', 'Оброблене зображення'])


# Task #7
images = [None] * 2
images[0] = original_image_7
images[1] = mtf.watershed_with_grad(original_image_7)
mgpf.show_images([1,2], images, win_name_task_7, ['Оригінальне зображення', 'Оброблене зображення'])

