import numpy as np
import cv2
from matplotlib import pyplot as plt

def low_pass_filter(image, cutoff_frequency):
    # Застосування двовимірного перетворення Фур'є
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Отримання розміру зображення
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Створення маски з низькочастотними характеристиками
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[center_row - cutoff_frequency:center_row + cutoff_frequency,
         center_col - cutoff_frequency:center_col + cutoff_frequency] = 1

    # Застосування маски до зсунутого частотного представлення
    f_transform_shifted_filtered = f_transform_shifted * mask

    # Зворотне перетворення Фур'є для отримання зображення у просторовій області
    image_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(f_transform_shifted_filtered)))

    return image_filtered

# Зчитання зображення
image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

# Застосування фільтра низьких частот із частотою відсічення 30 пікселів
cutoff_frequency = 30
filtered_image = low_pass_filter(image, cutoff_frequency)

# Відображення оригінального та обробленого зображень
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Оригінальне зображення')
plt.axis('off')
plt.subplot(122), plt.imshow(filtered_image, cmap='gray'), plt.title('Оброблене зображення')
plt.axis('off')
plt.show()
