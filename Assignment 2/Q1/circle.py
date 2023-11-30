import cv2
import numpy as np

def are_circles_detected(image_path, min_radius=10, max_radius=50):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and help circle detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Use HoughCircles to detect circles in the image
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # Return True if circles are found, False otherwise
    return circles is not None

# Replace 'your_image_path.jpg' with the path to your image file
circles_detected = are_circles_detected('test/lava21.jpg')

if circles_detected:
    print("Circles detected in the image!")
else:
    print("No circles detected in the image.")
