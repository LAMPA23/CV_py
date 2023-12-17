import cv2
import numpy as np


def BGR_to_CMY(image):
    blue, green, red = cv2.split(image)
    cyan = 255 - red
    magenta = 255 - green
    yellow = 255 - blue
    return cv2.merge([cyan, magenta, yellow])


def BGR_to_HSI(image):
    image_normalized = image / 255.0
    blue, green, red = cv2.split(image_normalized)
    intensity = (red + green + blue) / 3.0
    denominator = np.sqrt((red - green)**2 + (red - blue) * (green - blue))
    denominator[denominator == 0] = 1e-5  # To avoid division by zero
    angle = np.arccos(0.5 * ((red - green) + (red - blue)) / denominator)
    hue = angle.copy()
    hue[blue > green] = 2 * np.pi - hue[blue > green]
    saturation = 1 - 3 * np.minimum(red, np.minimum(green, blue)) / (red + green + blue)
    hue = hue * 180 / np.pi
    return cv2.merge([hue, saturation, intensity])