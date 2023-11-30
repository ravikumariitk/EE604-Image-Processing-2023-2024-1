import cv2 as cv
import numpy as np

def smooth(image):
    b, g, r = cv.split(image)
    b_blurred = cv.GaussianBlur(b, (0, 0), 3)
    g_blurred = cv.GaussianBlur(g, (0, 0), 3)
    r_blurred = cv.GaussianBlur(r, (0, 0), 3)
    b_unsharp = cv.addWeighted(b, 1.5, b_blurred, -0.5, 0)
    g_unsharp = cv.addWeighted(g, 1.5, g_blurred, -0.5, 0)
    r_unsharp = cv.addWeighted(r, 1.5, r_blurred, -0.5, 0)
    sharpened = cv.merge((b_unsharp, g_unsharp, r_unsharp))
    return sharpened

def amplify_image(image):
    scale_factor = 2
    resized_image = cv.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)
    amplified_image = cv.convertScaleAbs(resized_image, alpha=1.5, beta=0.0)
    return amplified_image

def face_detection(cartoon_image):
    grayscale_image = cv.cvtColor(cartoon_image, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(grayscale_image)
    for (x, y, w, h) in detected_faces:
        cv.rectangle(cartoon_image,(x, y),(x + w, y + h),(0, 255, 0),2)
    return len(detected_faces)

def solution(image_path):
    image=cv.imread(image_path)
    image=smooth(image)
    image=amplify_image(image)
    total_faces=face_detection(image)
    if total_faces == 10:
        height, width, _ = image.shape
        half_width = width // 2
        left_half = image[:, :half_width, :]
        right_half = image[:, half_width:, :]
        right=face_detection(right_half)
        left= face_detection(left_half)
        if right==5 and left==4 :
            return 'real'
        else :
            return 'fake'
    return 'fake'