import cv2
import numpy as np

# Load the image
image = cv2.imread('test/3_e.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Find lines using Hough Line Transform
lines = cv2.HoughLines(binary, 1, np.pi / 180, 200)

if lines is not None:
    # Calculate the average angle of detected lines
    angles = []
    for rho, theta in lines[:, 0]:
        angles.append(theta)

    # Calculate the median angle as it is more robust to outliers
    median_angle = np.median(angles)

    # Convert the angle to degrees
    rotation_angle = median_angle * 180 / np.pi

    # Get the image dimensions
    height, width = image.shape[:2]

    # Create a rotation matrix to perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -rotation_angle, 1)

    # Rotate the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    # Display the rotated image
    cv2.imshow("Rotated Image", rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()