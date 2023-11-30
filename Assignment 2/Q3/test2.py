import cv2



def face_detection(cartoon_image):
    # grayscale_image = cv2.cvtColor(cartoon_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(cartoon_image)
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(cartoon_image,(x, y),(x + w, y + h),(0, 255, 0),2)
    cv2.imshow("faces",cartoon_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return len(detected_faces)

# Read the image
img = cv2.imread('test/r4.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a median blur to the image
median_blurred = cv2.medianBlur(gray, 5)

# Apply a histogram equalization to the image
hist_equ = cv2.equalizeHist(median_blurred)

# Apply a color thresholding to the image
thresh = cv2.threshold(hist_equ, 128, 255, cv2.THRESH_BINARY_INV)[1]
face_detection(thresh)