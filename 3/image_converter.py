import cv2
import numpy as np
from PIL import Image

def get_color_space(image_path):
    return Image.open(image_path).mode


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


def to_smooth(image, sigma=1.0):
    red_channel, green_channel, blue_channel = cv2.split(image)
    smooth_blue = cv2.GaussianBlur(blue_channel, (0, 0), sigma)
    smooth_green = cv2.GaussianBlur(green_channel, (0, 0), sigma)
    smooth_red = cv2.GaussianBlur(red_channel, (0, 0), sigma)
    return cv2.merge([smooth_red, smooth_green, smooth_blue])
