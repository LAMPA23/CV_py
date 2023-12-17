import cv2
import numpy as np
import os


def show_image(image_src, title='some image'):
    image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    cv2.imshow(title, image_src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_images(images, titles):
    if len(images) != len(titles):
        print('Error len(images) != len(titles)')
        exit()
    for cnt in range(len(images)):
        cv2.imshow(titles[cnt], images[cnt])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(image, title='some image', dir='mdf_images', dir_path='C:/Users/bimee/OneDrive/computer vision/3/'):
    os.chdir(dir_path)
    if dir not in os.listdir():
        os.makedirs(dir)
    cv2.imwrite(f'{dir_path}/{dir}/{title}.png', image)


def show_and_save_image(image, title='some image', dir='mdf_images', dir_path='C:/Users/bimee/OneDrive/computer vision/3/'):
    os.chdir(dir_path)
    if dir not in os.listdir():
        os.makedirs(dir)
    cv2.imwrite(f'{dir_path}/{dir}/{title}.png', image)
    show_image(image, title)
