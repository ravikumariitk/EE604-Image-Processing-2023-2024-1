import cv2
import numpy as np

def unsharp_masking_colored(image_path, sigma=1.0, strength=1.5):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Apply GaussianBlur to create a blurred version
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)

    # Convert the image to float32 for better precision during calculations
    img_float32 = img.astype(np.float32)
    blurred_float32 = blurred.astype(np.float32)

    # Calculate the unsharp mask for each channel
    unsharp_mask = img_float32 - strength * (img_float32 - blurred_float32)

    # Clip the values to the valid range [0, 255]
    unsharp_mask = np.clip(unsharp_mask, 0, 255)

    # Convert the result back to 8-bit unsigned integer for display
    unsharp_mask_colored = unsharp_mask.astype(np.uint8)

    # Display the original, blurred, and unsharp masked images
    cv2.imshow('Original Image', img)
    cv2.imshow('Blurred Image', blurred)
    cv2.imshow('Unsharp Masked Image', unsharp_mask_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'your_image_path.jpg' with the path to your image file
unsharp_masking_colored('output.jpg')
