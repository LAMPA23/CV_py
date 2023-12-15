import sys
import dftfilter
import cv2
import numpy as np


def get_originals_param(image_path):
    origin_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2GRAY)
    origin_PQ = dftfilter.paddedsize(origin_image)
    origin_FFT2 = np.fft.fft2(origin_image, origin_PQ)
    return origin_image, origin_PQ, origin_FFT2


def arg_check(filter_type, pass_type, filter_radius):
    possible_filter_type = ["ideal", "gaussian", "btw", "laplacian"]
    possible_pass_type = ["hp", "lp"]
    if filter_type not in possible_filter_type:
        print(f'Error in arg_check(). The "filter_type" not can be {filter_type}')
        sys.exit()
    if pass_type not in possible_pass_type:
        print(f'Error in arg_check(). The "pass_type" not can be {pass_type}')
        sys.exit()
    if pass_type == 'lp' and filter_type == 'laplacian':
        print(f'Error in arg_check(). The "filter_type" not can be {filter_type} when pass_type = {pass_type}')
        sys.exit()
    if filter_radius <= 0:
        print(f'Error in arg_check(). filter_radius mast be > 0')
        sys.exit()


def get_filter_function(origin_PQ, filter_type, pass_type, filter_radius = 2, but_filt_ord = 2):
    arg_check(filter_type, pass_type, filter_radius)
    if pass_type == "hp":
        filter_function = dftfilter.hp_filter(filter_type, origin_PQ, filter_radius, but_filt_ord)
    else:
        filter_function = dftfilter.lp_filter(filter_type, origin_PQ, filter_radius, but_filt_ord)
    return filter_function


def get_filter_image(origin_PQ, origin_FFT2, filter_function):
    new_image = np.real(np.fft.ifft2(filter_function * origin_FFT2))
    return new_image[0 : (origin_PQ[0] // 2), 0 : (origin_PQ[1] // 2)]


def get_filter_mask(filter_function):
    return np.fft.fftshift(filter_function)




def get_equalized_image(path_image):
    gray_image = cv2.cvtColor(cv2.imread(path_image), cv2.COLOR_RGB2GRAY)
    hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    equalized_image = np.interp(gray_image.flatten(), bins[:-1], cdf_normalized)
    return equalized_image.reshape(gray_image.shape).astype(np.uint8)   

def get_histograme(gray_image):
    hist, _ = np.histogram(gray_image.flatten(), 256, [0, 256])
    return hist