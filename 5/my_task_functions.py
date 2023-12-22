import os 
import cv2
import numpy as np
import skimage.morphology as morphology
import skimage.measure as measure


def detect_points(image, threshold_fraction=0.1):
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
    # Set the threshold for point detection
    threshold = threshold_fraction * np.max(np.abs(laplacian))
    
    # Create a mask for the points
    points_mask = np.zeros_like(laplacian, dtype=np.uint8)
    points_mask[np.abs(laplacian) >= threshold] = 255
    
    # Resize the Laplacian and the points mask to match the original image size
    laplacian_resized = cv2.resize(laplacian, (image.shape[1], image.shape[0]))
    points_mask_resized = cv2.resize(points_mask, (image.shape[1], image.shape[0]))
    
    # Convert the points mask to a color image for overlay
    points_overlay = cv2.cvtColor(points_mask_resized, cv2.COLOR_GRAY2BGR)
    
    # Overlay the mask on the original image
    result_image = cv2.addWeighted(image, 0.7, points_overlay, 0.3, 0)
    
    return result_image


