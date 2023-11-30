import cv2
import numpy as np

def background_remove(image):
    rect = (0, 0, image.shape[1], image.shape[0]-1)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 2) | (mask == 0), 0, -1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    return result

def kmean(image):
    b, g, r = cv2.split(image)
    blue_channel = b.reshape(-1, 1).astype(np.float32)
    K = 2
    kmeans = cv2.kmeans(blue_channel, K, None, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    centers = kmeans[2]
    cluster_centers = np.uint8(centers)
    high_intensity_mask = (b > cluster_centers[1]).astype(np.uint8) * 255
    return high_intensity_mask

def combine_clusters(image):
    binary_image = image
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = np.zeros_like(binary_image)
    cv2.drawContours(result_image, contours, -1, 255, thickness=-1)
    return result_image

def transform(image):
    mask = image > 0
    image[mask] = 255
    return image

def convert(binary_image):
    binary_image_3_channel = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        binary_image_3_channel[:, :, i] = binary_image
    return binary_image_3_channel

def solution(image_path):
    image = cv2.imread(image_path)
    image = background_remove(image)
    image = kmean(image)
    image = combine_clusters(image)
    image = transform(image)
    image = convert(image)
    return image