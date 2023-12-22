import os 
import cv2
import numpy as np
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
original_image_rice = mf.get_gray_image(kernal_path, 'rice.png')
original_image_cameraman = mf.get_gray_image(kernal_path, 'cameraman.tif')


# Task #1 Дилатація (файл – pic.1.jpg)
binary_dilated = morphology.binary_dilation(original_image_1)
mf.show_images_x2(
    original_image_1,
    binary_dilated,
    'Task #1 Дилатація (файл – pic.1.jpg)',
    'original_image_1',
    'binary_dilated')


# Task #2 Ерозія (файл – pic.2.jpg)
binary_erosion = morphology.binary_erosion(original_image_2)
mf.show_images_x2(
    original_image_2,
    binary_erosion,
    'Task #2 Ерозія (файл – pic.2.jpg)',
    'original_image_2',
    'binary_erosion')


# Task #3 Розмикання та замикання (файл – pic.3.jpg)
binary_image = cv2.threshold(original_image_3, 128, 255, cv2.THRESH_BINARY)[1]
binary_opening = morphology.binary_opening(binary_image)
binary_closing = morphology.binary_closing(binary_image)
mf.show_images_x3(
    original_image_3,
    binary_opening,
    binary_closing,
    'Task #3 Розмикання та замикання (файл – pic.3.jpg)',
    'original_image_3',
    'binary_opening',
    'binary_closing')


# Task #4 Потоншення (файл – pic.3.jpg)
thin_image = morphology.thin(original_image_3)
mf.show_images_x2(
    original_image_3,
    thin_image,
    'Task #4 Потоншення (файл – pic.3.jpg)',
    'original_image_3',
    'thin_image')


# Task #5 Побудова остова (файл – pic.4.jpg)
binary_image = cv2.threshold(original_image_4, 128, 255, cv2.THRESH_BINARY)[1]
skeleton_image = morphology.skeletonize(binary_image)
mf.show_images_x2(
    original_image_4,
    skeleton_image,
    'Task #5 Побудова остова (файл – pic.4.jpg)',
    'original_image_4',
    'skeleton_image')


# Task #6 Виділення компонент зв’язності (файл – pic.5.jpg)
binary_image = cv2.threshold(original_image_5, 128, 255, cv2.THRESH_BINARY)[1]
labeled_image = measure.label(binary_image)
mf.show_images_x2(
    original_image_5,
    labeled_image,
    'Task #6 Виділення компонент зв’язності (файл – pic.5.jpg)',
    'original_image_5',
    f'labeled_image {np.max(original_image_5)}')


# Task #7 Морфологічна реконструкція (слайд – 14, файли – pic.6a.tif та pic.6b.tif)
binary_image_a = cv2.threshold(original_image_6a, 128, 255, cv2.THRESH_BINARY)[1]
binary_image_b = cv2.threshold(original_image_6b, 128, 255, cv2.THRESH_BINARY)[1]
reconstruction = morphology.reconstruction(binary_image_b, binary_image_a)
mf.show_images_x3(
    binary_image_a,
    binary_image_b,
    reconstruction,
    'Task #7 Морфологічна реконструкція (слайд – 14, файли – pic.6a.tif та pic.6b.tif)',
    'binary_image_a',
    'binary_image_b',
    'reconstruction')


# Task #8 Морфологічна реконструкція (слайд – 15, файл – pic.7.tif)
_, binary_image = cv2.threshold(original_image_7, 128, 255, cv2.THRESH_BINARY)
seed_image = np.zeros_like(binary_image, dtype=np.uint8)
reconstruction = morphology.reconstruction(seed_image, binary_image)
mf.show_images_x2(
    original_image_7,
    reconstruction,
    'Task #8 Морфологічна реконструкція (слайд – 15, файл – pic.7.tif)',
    'original_image_7',
    'reconstruction')


# Task #9 Півтонова дилатація та ерозія (файл – pic.8.tif)
dilated_image = morphology.dilation(original_image_8)
erosion_image = morphology.erosion(original_image_8)
mf.show_images_x3(
    original_image_8,
    dilated_image,
    erosion_image,
    'Task #9 Півтонова дилатація та ерозія (файл – pic.8.tif)',
    'original_image_8',
    'dilated_image',
    'erosion_image')


# Task #10 Півтонове розмикання та замикання (файл – pic.9.tif)
dilated_image = morphology.dilation(original_image_9)
erosion_image = morphology.erosion(original_image_9)
mf.show_images_x3(
    original_image_9,
    dilated_image,
    erosion_image,
    'Task #10 Півтонове розмикання та замикання (файл – pic.9.tif)',
    'original_image_9',
    'dilated_image',
    'erosion_image')


# Task #11 Морфологічний градієнт (файл – cameraman.tif)
dilated_image = morphology.dilation(original_image_cameraman)
erosion_image = morphology.erosion(original_image_cameraman)
gradient = dilated_image - erosion_image
mf.show_images_x4(
    original_image_cameraman,
    dilated_image,
    erosion_image,
    gradient,
    'Task #11 Морфологічний градієнт (файл – cameraman.tif)',
    'original_image_cameraman',
    'dilated_image',
    'erosion_image',
    'gradient')


# Task #12 Перетворення «виступ» (файл – rice.png)
opening_image = morphology.opening(original_image_rice, morphology.square(7))
tophat_transform = original_image_rice - opening_image
mf.show_images_x3(
    original_image_rice,
    opening_image,
    tophat_transform,
    'Task #12 Перетворення «виступ» (файл – rice.png)',
    'original_image_rice',
    'opening_image',
    'tophat_transform')


# Task #13 Морфологічна півтонова реконструкція (файл – pic.10.tif)
original_image_10 = original_image_10 // 2
_, binary_image = cv2.threshold(original_image_10, 128, 255, cv2.THRESH_BINARY)
reconstructed_image = morphology.reconstruction(seed=binary_image, mask=original_image_10)
mf.show_images_x2(
    original_image_10 * 2,
    reconstructed_image,
    'Task #13 Морфологічна півтонова реконструкція (файл – pic.10.tif)',
    'original_image_10',
    'reconstructed_image')
