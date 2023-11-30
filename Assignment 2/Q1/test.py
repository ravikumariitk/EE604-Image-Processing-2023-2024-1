import cv2
import numpy as np
from skimage import io, exposure
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square

def edge(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 300)
    return edges

def binary(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = exposure.adjust_gamma(gray_image, gamma=2)
    threshold_value = threshold_otsu(enhanced_image)
    binary_mask = enhanced_image > threshold_value
    binary_mask = closing(binary_mask, square(3))
    lava_binary = np.zeros_like(image)
    lava_binary[binary_mask] = 255
    cv2.imshow("Binary mask", lava_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return lava_binary

def solve(image):
    rect = (0, 0, image.shape[1], image.shape[0]-1)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    cv2.grabCut(image, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 2) | (mask == 0), 0, -1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    return result

def transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    image[mask] = 255
    return image

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Read the image
image = cv2.imread('test/lava20.jpg')
image=solve(image)
cv2.imshow('Image without Sky',image)
cv2.waitKey(0)



# Split the image into its color channels (B, G, R)
b, g, r = cv2.split(image)

# Create a feature vector containing only the blue channel
blue_channel = b.reshape(-1, 1).astype(np.float32)

# Define the number of clusters (2 for dim and high intensity)
K = 2

# Apply k-means clustering
kmeans = cv2.kmeans(blue_channel, K, None, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

# Get the cluster centers
centers = kmeans[2]

# Reshape the cluster centers to match the original image shape
cluster_centers = np.uint8(centers)

# Create a mask for the high-intensity blue pixels
high_intensity_mask = (b > cluster_centers[1]).astype(np.uint8) * 255

# Create a mask for the dim blue pixels
dim_intensity_mask = (b <= cluster_centers[1]).astype(np.uint8) * 255

# Display or save the result
cv2.imshow("High Intensity (Blue)", high_intensity_mask)
cv2.imshow("Dim Intensity (Blue)", dim_intensity_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# import cv2
# import numpy as np

# Load the binary image
binary_image =high_intensity_mask

# Apply morphological operations to combine small clusters
kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# Find contours in the dilated image
contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty black image
result_image = np.zeros_like(binary_image)

# Draw the outer contour containing all small clusters
cv2.drawContours(result_image, contours, -1, 255, thickness=-1)

# Display or save the result
cv2.imshow("Combined Clusters", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
