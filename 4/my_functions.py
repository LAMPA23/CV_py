import cv2
import matplotlib.pyplot as plt

def show_images_x2(
    image_1,
    image_2,
    window_name,
    title_1,
    title_2
    ):

    fig, _ = plt.subplots(1, 2)
    fig.canvas.manager.set_window_title(window_name)

    plt.subplot(1,2,1)
    plt.imshow(image_1, cmap='gray')
    plt.title(title_1)
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(image_2, cmap='gray')
    plt.title(title_2)
    plt.axis('off')

    plt.show()


def show_images_x3(
    image_1,
    image_2,
    image_3,
    window_name,
    title_1,
    title_2,
    title_3
    ):

    fig, _ = plt.subplots(1, 3)
    fig.canvas.manager.set_window_title(window_name)

    plt.subplot(1,3,1)
    plt.imshow(image_1, cmap='gray')
    plt.title(title_1)
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.imshow(image_2, cmap='gray')
    plt.title(title_2)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(image_3, cmap='gray')
    plt.title(title_3)
    plt.axis('off')

    plt.show()


def show_images_x4(
    image_1,
    image_2,
    image_3,
    image_4,
    window_name,
    title_1,
    title_2,
    title_3,
    title_4
    ):

    fig, _ = plt.subplots(2, 2)
    fig.canvas.manager.set_window_title(window_name)

    plt.subplot(2,2,1)
    plt.imshow(image_1, cmap='gray')
    plt.title(title_1)
    plt.axis('off')
    
    plt.subplot(2,2,2)
    plt.imshow(image_2, cmap='gray')
    plt.title(title_2)
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(image_3, cmap='gray')
    plt.title(title_3)
    plt.axis('off')
    
    plt.subplot(2,2,4)
    plt.imshow(image_4, cmap='gray')
    plt.title(title_4)
    plt.axis('off')

    plt.show()


def get_gray_image(kernal_path, image_name):
    original_image = cv2.imread(f'{kernal_path}{image_name}')
    return cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)