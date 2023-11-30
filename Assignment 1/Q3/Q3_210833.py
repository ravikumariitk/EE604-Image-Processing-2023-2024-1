import cv2
import numpy as np
import math
from sklearn.linear_model import LinearRegression

# Function to find fft
def fft(image):
    # Computing FFT
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)
    # Applying window to reduce the defects
    window = np.hamming(fft_shifted.shape[0])
    window_reshaped = np.expand_dims(window, axis=1)
    magnitude_spectrum = np.log(np.abs(fft_shifted * window_reshaped) + 1)
    # Normalization
    magnitude_spectrum_norm = np.uint8(cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX))
    #returning FFT of image
    return magnitude_spectrum_norm

# Function to get the slope of the best fit line in the fft of image
def find_slope(image):
  gray_image = fft(image)
  # Applying thresholding to highlight the points
  threshold_value = 180
  thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)[1]
  height,width=thresholded_image.shape[:2]
  # Finding slope of best fit line using linear regression
  points=[]
  for i in range(0,height):
    for j in range(0,width):
      if thresholded_image[i][j] !=0 :
        points.append([j-int(width/2),i-int(height/2)])
  X = []
  Y = []
  for i in range(1, len(points)):
    X.append(points[i][0])
    Y.append(points[i][1])
  X = np.array(X)
  Y = np.array(Y)
  varience=np.var(X)
  if varience < 1:
    return 0
  model = LinearRegression()
  model.fit(X.reshape(-1,1), Y)
  slope = model.coef_
  if slope<0:
    slope=-slope
    angle=math.atan(slope)
    angle=(angle*180)/math.pi
    angle=180-angle
  else:
    angle=math.atan(slope)
    angle=(angle*180)/math.pi
  return angle

def solution(image_path):
  img=cv2.imread(image_path)
  gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  # Angle of the rotation
  ang=find_slope(gray_img)
  height,width=img.shape[:2]
  center_img=(width/2,height/2)
  # Rotating the image @ ang using rotation matrix
  if(ang==0):
    return img
  rotationMatrix = cv2.getRotationMatrix2D(center_img,-(90-ang), 1.0)
  rotated_img = cv2.warpAffine(img, rotationMatrix, (width + 50, height), borderMode=cv2.BORDER_REPLICATE)
  return rotated_img