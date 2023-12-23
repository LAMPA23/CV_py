import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans_clustering(image_path, num_clusters):
    # Зчитування зображення
    image = cv2.imread(image_path)
    
    # Перетворення зображення до вигляду, придатного для kmeans
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # Використання kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Конвертація центрів кластерів у цілочисельний тип
    centers = np.uint8(centers)
    
    # Заміна кожного пікселя кольором його кластера
    segmented_image = centers[labels.flatten()]
    
    # Повернення результату у формі вихідного розміру зображення
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image

# Приклад використання
image_path = "original_images/pic.8.jpg"
num_clusters = 30  # Кількість кластерів
result = kmeans_clustering(image_path, num_clusters)

# Виведення результату
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('K-Means Clustering')
plt.show()
