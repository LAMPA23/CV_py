import cv2
import numpy as np
import colorgrad 
from PIL import Image



# Color space conversion functions

def get_color_space(image_path):
    return Image.open(image_path).mode

def RGB_to_CMY(image):
    blue, green, red = cv2.split(image)
    cyan = 255 - red
    magenta = 255 - green
    yellow = 255 - blue
    return cv2.merge([cyan, magenta, yellow])

def RGB_to_HSI(image):
    image_normalized = image / 255.0
    red, green, blue = cv2.split(image_normalized)
    intensity = (red + green + blue) / 3.0
    denominator = np.sqrt((red - green)**2 + (red - blue) * (green - blue))
    denominator[denominator == 0] = 1e-5  # To avoid division by zero
    angle = np.arccos(0.5 * ((red - green) + (red - blue)) / denominator)
    hue = angle.copy()
    hue[blue > green] = 2 * np.pi - hue[blue > green]
    saturation = 1 - 3 * np.minimum(red, np.minimum(green, blue)) / (red + green + blue)
    hue = hue * 180 / np.pi
    return cv2.merge([hue, saturation, intensity])






# Smoothing functions

def smooth_RGB(image, sigma=1.0):
    red_channel, green_channel, blue_channel = cv2.split(image)
    smooth_blue = cv2.GaussianBlur(blue_channel, (0, 0), sigma)
    smooth_green = cv2.GaussianBlur(green_channel, (0, 0), sigma)
    smooth_red = cv2.GaussianBlur(red_channel, (0, 0), sigma)
    return cv2.merge([smooth_red, smooth_green, smooth_blue])

def smooth_HSV(image, smoothing_factor=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:,:,2] = cv2.blur(image[:,:,2], (smoothing_factor,smoothing_factor))
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def smooth_Lab(image, smoothing_factor=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    image[:,:,0] = cv2.blur(image[:,:,0], (smoothing_factor,smoothing_factor))
    return cv2.cvtColor(image, cv2.COLOR_Lab2RGB)





# Increase sharpness functions

def increase_sharpness_RGB(image):
    sharpness_filater = np.array([
        [-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]
    ])
    return cv2.filter2D(image, -1, sharpness_filater)

def increase_sharpness_HSV(image):
    sharpness_filater = np.array([
        [ 0, 2, 0],
        [ 2,-8, 2],
        [ 0, 2, 0]
    ])
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)
    v_channel = cv2.subtract(v_channel, cv2.filter2D(v_channel, -1, sharpness_filater))
    hsv_image_sharp = cv2.merge([h_channel, s_channel, v_channel])
    return cv2.cvtColor(hsv_image_sharp, cv2.COLOR_HSV2RGB)

def increase_sharpness_Lab(image):
    sharpness_filater = np.array([
        [ 0, 1, 0],
        [ 1,-4, 1],
        [ 0, 1, 0]
    ])
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    l_channel = lab_image[:,:,0]
    sharpness_l_channel = cv2.subtract(l_channel, cv2.filter2D(l_channel, -1, sharpness_filater))
    lab_image[:,:,0] = sharpness_l_channel
    return cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB)





# Equalization functions

def equaliz_RGB(image):
    red_channel, green_channel, blue_channel = cv2.split(image)
    equalized_red_channel = cv2.equalizeHist(red_channel)
    equalized_green_channel = cv2.equalizeHist(green_channel)
    equalized_blue_channel = cv2.equalizeHist(blue_channel)
    return cv2.merge((equalized_red_channel, equalized_green_channel, equalized_blue_channel))

def equaliz_HSV(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(image)
    equalized_s_channel = cv2.equalizeHist(s_channel)
    equalized_v_channel = cv2.equalizeHist(v_channel)
    image = cv2.merge((h_channel, equalized_s_channel, equalized_v_channel))
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def equaliz_Lab(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    l_channel, a_channel, b_channel = cv2.split(image)
    equalized_l_channel = cv2.equalizeHist(l_channel)
    image = cv2.merge((equalized_l_channel, a_channel, b_channel))
    return cv2.cvtColor(image, cv2.COLOR_Lab2RGB)




# Gradienting functions

def get_per_plane_gradient(image):
    _, _, PPG = colorgrad.colorgrad(image)
    return PPG

def get_sobel_gradient_normalized(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_gradient = cv2.magnitude(sobel_x, sobel_y)
    return cv2.normalize(sobel_gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

def get_absolute_difference(image):
    PPG = get_per_plane_gradient(image)
    sobel_gradient_normalized = get_sobel_gradient_normalized(image)
    return cv2.absdiff(sobel_gradient_normalized.astype(PPG.dtype), PPG)
