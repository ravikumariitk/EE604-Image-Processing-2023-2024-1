import numpy as np
import cv2

def create_window(i,low,high, sigma):
    start = int(i - 2 * sigma)
    stop = int(i + 2 * sigma + 1)
    result_array = np.arange(start, stop)
    for i in range(len(result_array)):
        if low > min(result_array[i], high):
            result_array[i] = low
        else:
            result_array[i] = max(low, min(high, result_array[i]))
    return result_array

def bilateral_filter(image, spatial_sigma, intensity_sigma):
    height, width, channels = image.shape
    filtered_image = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            spatial_window = create_window(i , 0 , height-1 , spatial_sigma)
            intensity_window = create_window(j , 0 , width-1 , spatial_sigma)
            spatial_weights = np.exp(-((i - spatial_window)**2) / (2 * spatial_sigma**2))
            intensity_weights = np.exp(-((j - intensity_window)**2) / (2 * intensity_sigma**2))
            bilateral_weights = np.outer(spatial_weights, intensity_weights)
            normalized_weights = bilateral_weights / np.sum(bilateral_weights)
            selected_portion = image[spatial_window, intensity_window, :]
            selected_portion = image[spatial_window, intensity_window, :]
            weighted_portion = normalized_weights[:, :, None] * selected_portion
            weighted_sum = np.sum(weighted_portion, axis=(0, 1))
            filtered_image[i, j, :] = weighted_sum
    return filtered_image.astype(np.uint8)

def cross_bilateral_filter(flash_image, no_flash_image, alpha):
    filtered_flash = bilateral_filter(flash_image, 1, 100)
    filtered_no_flash = bilateral_filter(no_flash_image, 1, 100)
    prolight_image = alpha * filtered_flash + (1 - alpha) * filtered_no_flash
    return prolight_image

def solution(image_path_a, image_path_b):
    alpha = 0.1
    flash_image = cv2.imread(image_path_b)
    non_flash_image = cv2.imread(image_path_a)
    image = cross_bilateral_filter(flash_image, non_flash_image, alpha)
    return image