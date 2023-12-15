import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.system('cls')

def equalized_image(image, normalize_range):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * normalize_range / cdf[-1]
    equalized_image = np.interp(gray_image.flatten(), bins[:-1], cdf_normalized)
    equalized_image = equalized_image.reshape(gray_image.shape).astype(np.uint8)
    return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)   

def show_histograme(image, plt_color, plt_xlable, plt_ylable, plt_title):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
    plt.figure(figsize=(4, 3))
    plt.plot(hist, color=plt_color)
    plt.xlabel(plt_xlable)
    plt.ylabel(plt_ylable)
    plt.title(plt_title)
     

original_image = cv2.imread('pic4.jpg')

full_equalized_image = equalized_image(original_image, 255)
part_equalized_image = equalized_image(original_image, 180)

show_histograme(original_image, 'black', 'Інтенсивність пікселя', 'Кількість пікселів такої інтенсивності', 'Оигінальне зображення')
show_histograme(full_equalized_image, 'red', 'Інтенсивність пікселя', 'Кількість пікселів такої інтенсивності', 'Повністю еквалізоване зображення')
show_histograme(part_equalized_image, 'orange', 'Інтенсивність пікселя', 'Кількість пікселів такої інтенсивності', 'Частково еквалізоване зображення')

cv2.imshow('Original Image', original_image)
cv2.imshow('FULL Equalized Image', full_equalized_image)
cv2.imshow('PART Equalized Image', part_equalized_image)
plt.show()
cv2.destroyAllWindows()
