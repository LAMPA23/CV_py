import cv2
import numpy

from skimage import feature, segmentation, filters, util
from scipy import ndimage

import masks



def imfilter(image, mask = masks.laplacian):
    laplacian_result = numpy.zeros_like(image, dtype=numpy.float32)
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            laplacian_result[y, x] = max(0,numpy.sum(image[y-1:y+2, x-1:x+2] * mask))
    return cv2.convertScaleAbs(laplacian_result)




def detect_points(image, threshold_fraction=0.9):
    # apply laplacian filtration
    laplacian_image = imfilter(image)

    # find biggest value
    threshold_value = numpy.max(laplacian_image) * threshold_fraction

    # zerofication small value
    for y in range(laplacian_image.shape[0]):
        for x in range(laplacian_image.shape[1]):
            if laplacian_image[y,x] < threshold_value:
                laplacian_image[y,x] = 0
            else:
                laplacian_image[y,x] = 255

    return laplacian_image

    


def line_finder(image):
    vertical = imfilter(image, masks.vertical)
    horizontal = imfilter(image, masks.horizontal)
    diagonal_right = imfilter(image, masks.diagonal_right)
    diagonal_left = imfilter(image, masks.diagonal_left)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            image[y,x] = (vertical[y,x] + horizontal[y,x] + diagonal_right[y,x] + diagonal_left[y,x]) % 255

    return [vertical, horizontal, diagonal_right, diagonal_left, image]


def watershed_with_distance(image):
    distance = ndimage.distance_transform_edt(image)
    coords = feature.peak_local_max(distance, footprint=numpy.ones((3, 3)), labels=image)
    mask = numpy.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    return segmentation.watershed(-distance, markers, mask=image)

    


def watershed_with_grad(image):
    sobelxy = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=2, dy=2, ksize=3)
    coords = feature.peak_local_max(sobelxy, footprint=numpy.ones((3, 3)), labels=image)
    mask = numpy.zeros(sobelxy.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    water = segmentation.watershed(image=-image, markers=markers, mask=sobelxy)
    return water



def kmeans_clustering(image, num_clusters):
    pixels = image.reshape((-1, 3))
    pixels = numpy.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = numpy.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image