import cv2
import numpy as np
def amplify_image(image):
    scale_factor = 2
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    amplified_image = cv2.convertScaleAbs(resized_image, alpha=1.0, beta=0.0)
    return amplified_image

def face_detection(cartoon_image):
    grayscale_image = cv2.cvtColor(cartoon_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    detected_faces = face_cascade.detectMultiScale(grayscale_image)
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(cartoon_image,(x, y),(x + w, y + h),(0, 255, 0),2)
    cv2.imshow("faces",cartoon_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return len(detected_faces)

def smooth(image):
    # Split the image into its color channels (BGR)
    b, g, r = cv2.split(image)

    # Apply Gaussian blur to each color channel
    b_blurred = cv2.GaussianBlur(b, (0, 0), 3)
    g_blurred = cv2.GaussianBlur(g, (0, 0), 3)
    r_blurred = cv2.GaussianBlur(r, (0, 0), 3)

    # Calculate the unsharp masks for each channel
    b_unsharp = cv2.addWeighted(b, 1.5, b_blurred, -0.5, 0)
    g_unsharp = cv2.addWeighted(g, 1.5, g_blurred, -0.5, 0)
    r_unsharp = cv2.addWeighted(r, 1.5, r_blurred, -0.5, 0)

    # Merge the color channels back together
    sharpened = cv2.merge((b_unsharp, g_unsharp, r_unsharp))
    return sharpened

def edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return edges


def edge1(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    mask = np.zeros_like(image)
    mask[edges != 0] = [0,0,0]
    darkened_image = cv2.add(image, mask)
    return darkened_image


def log_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    log_transformed = np.log1p(gray)
    log_transformed = (log_transformed / log_transformed.max()) * 255
    log_transformed = log_transformed.astype(np.uint8)
    return log_transformed



def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gamma(image):
    gamma = 0.5
    gamma_corrected = np.power(image / 255.0, 1.0 / gamma)
    gamma_corrected = (gamma_corrected * 255.0).astype(np.uint8)
    return gamma_corrected


def watershed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Find sure foreground using distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Subtract sure foreground from sure background to get unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all labels so that sure background is 0, not 1
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply the watershed algorithm
    cv2.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]  # Mark the boundaries with red color
    return image

image=cv2.imread('test/r4.jpg')
image=watershed(image)
# image=amplify_image(smooth(image))
# # image=gray(image)
# image=gamma(image)
# image=edge1(image)
cv2.imshow("output",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(face_detection(image))