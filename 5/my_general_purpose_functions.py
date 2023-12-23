import cv2
import matplotlib.pyplot as plt


def show_images(yx_size, images, window_name, titles):
    fig, _ = plt.subplots(yx_size[0], yx_size[1])
    fig.canvas.manager.set_window_title(window_name)

    for cnt in range(yx_size[0] * yx_size[1]):
        plt.subplot(yx_size[0], yx_size[1], cnt + 1)
        plt.imshow(images[cnt], cmap='gray')
        plt.title(titles[cnt])
        plt.axis('off')

    plt.show()


def get_gray_image(kernal_path, image_name):
    original_image = cv2.imread(f'{kernal_path}{image_name}')
    return cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)



