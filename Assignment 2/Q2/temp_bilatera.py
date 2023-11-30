import numpy as np
import cv2
import numpy as np

def bilateral_filter_color(image, spatial_sigma, intensity_sigma):
    height, width, channels = image.shape
    filtered_image = np.zeros_like(image, dtype=np.float32)
    spatial_weights = np.exp(-(np.arange(-2*spatial_sigma, 2*spatial_sigma+1)**2) / (2 * spatial_sigma**2))
    for i in range(height):
        for j in range(width):
            # Define the spatial and intensity neighborhoods
            spatial_window = np.arange(i - 2 * spatial_sigma, i + 2 * spatial_sigma + 1)
            intensity_window = np.arange(j - 2 * spatial_sigma, j + 2 * spatial_sigma + 1)
            # Ensure that the neighborhoods are within the image boundaries
            spatial_window = np.clip(spatial_window, 0, height - 1)
            intensity_window = np.clip(intensity_window, 0, width - 1)

            # Compute the Gaussian weights for the spatial and intensity neighborhoods
            spatial_weights_subset = np.exp(-((i - spatial_window)**2) / (2 * spatial_sigma**2))
            intensity_weights_subset = np.exp(-((j - intensity_window)**2) / (2 * intensity_sigma**2))

            # Compute the bilateral filter response
            bilateral_weights = np.outer(spatial_weights_subset, intensity_weights_subset)
            normalized_weights = bilateral_weights / np.sum(bilateral_weights)

            # Apply the filter to the pixel using vectorized operations
            filtered_image[i, j, :] = np.sum(normalized_weights[:,:,None] * image[spatial_window, intensity_window, :], axis=(0,1))

    return filtered_image.astype(np.uint8)

# Example usage:
# filtered = bilateral_filter_color(image, spatial_sigma=1, intensity_sigma=1)

# Example usage
image=cv2.imread('ultimate_test/2_a.jpg')
filtered = bilateral_filter_color(image, spatial_sigma=1, intensity_sigma=1e5)
cv2.imshow("Filtered", filtered)
cv2.waitKey(0)