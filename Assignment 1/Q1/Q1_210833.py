import cv2
import numpy as np
import math

# orange->172
# white->225
# green->75

# Function for getting uniques elements in array without lossing relative order
def get_unique(original_array):
    unique_set = set()
    unique_array = []
    for i in original_array:
        if i not in unique_set and (i == 172 or i==75 or i==255):
            unique_set.add(i)
            unique_array.append(i)
    return unique_array

# Function to return correct flag
def Correct_flag():
    flag = np.zeros((600, 600, 3), dtype=np.uint8)
    # Filling the flag with white color
    for i in range(0,600):
        for j in range(0,600):
            for k in range(0,3):
                flag[i][j][k]=255
    # Filling with orange color
    flag[0:200, :] = (51, 153, 255)
    flag[200:399, :] = 255
    # Filling with green color
    flag[399:600, :] = (0, 128, 0)
    # Drawing mid blue circle
    cv2.circle(flag, (300,300), 100, (255, 0, 0), 2)
    # Scatching 24 blue stoke each at 15 degree angle
    for angle in range(0, 375, 15):
        x1 = round(299 + 99 * math.cos(math.radians(angle)))
        y1 = round(299 + 99 * math.sin(math.radians(angle)))
        cv2.line(flag, (x1, y1), (299,299), (255, 0, 0), 1)
    return flag

# Function to return reversed flag
def Reversed_flag():
    flag=Correct_flag()
    rotated_flag = cv2.rotate(flag, cv2.ROTATE_90_CLOCKWISE)
    rotated_flag = cv2.rotate(rotated_flag, cv2.ROTATE_90_CLOCKWISE)
    return rotated_flag

# Function to return right flag
def Orange_at_right_flag():
    flag=Correct_flag()
    rotated_flag = cv2.rotate(flag, cv2.ROTATE_90_CLOCKWISE)
    return rotated_flag

# Function to return left flag
def Orange_at_left_flag():
    flag=Correct_flag()
    rotated_flag = cv2.rotate(flag, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated_flag

def solution(image_path):
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height,width=grayscale_image.shape
    # Cheaking for top layer
    curr2 = grayscale_image[2]
    curr=[]
    for i in curr2:
        if i!=0:
            curr.append(i)
    h,w = grayscale_image.shape
    if len(curr) > w/2 :
        unique_elements = get_unique(curr)
        if len(unique_elements)==1:
            if unique_elements[0]==172:
                # Flag is in corrrect orientation
                return Correct_flag()
            elif unique_elements[0] == 75:
                # Flag is in revesed orientation
                return Reversed_flag()
        if len(unique_elements)==3:
            if unique_elements[0]==172 :
                # Flag is in left orientation
               return Orange_at_left_flag()
            elif unique_elements[0]==75:
                # Flag is in right orientation
               return Orange_at_right_flag()
    # Checking buttom layer
    curr2 = grayscale_image[height-3]
    curr=[]
    for i in curr2:
        if i!=0:
            curr.append(i)
    if len(curr)>w/2:
        unique_elements = get_unique(curr)
        if len(unique_elements)==1:
            if unique_elements[0]==172:
                # Flag is in reversed orientation
                return Reversed_flag()
            elif unique_elements[0] == 75:
                # Flag is in corrrect orientation
                return Correct_flag()
        if len(unique_elements)==3:
            if unique_elements[0]==172 :
                # Flag is in left orientation
                return Orange_at_left_flag()
            elif unique_elements[0]==75:
                # Flag is in rigth orientation
                return Orange_at_right_flag()
    # Checking for left layer
    curr2 = [row[2] for row in grayscale_image]
    curr=[]
    for i in curr2:
        if i!=0:
            curr.append(i)
    if len(curr)>h/2:
        unique_elements = get_unique(curr)
        if len(unique_elements)==1:
            if unique_elements[0]==172:
                # Flag is in left orientation
                return Orange_at_left_flag()
            elif unique_elements[0] == 75:
                # Flag is in rigth orientation
               return Orange_at_right_flag()
        if len(unique_elements)==3:
            if unique_elements[0]==172 :
                # Flag is in correct orientation
                return Correct_flag()
            elif unique_elements[0]==75:
                # Flag is in reversed orientation
                return Reversed_flag()
    #checking for right layer
    curr2 = [row[width-3] for row in grayscale_image]
    curr=[]
    for i in curr2:
        if i!=0:
            curr.append(i)
    if len(curr)>h/2:
        unique_elements = get_unique(curr)
        if len(unique_elements)==1:
            if unique_elements[0]==172:
                # Flag is in right orientation
                return Orange_at_right_flag()
                return
            elif unique_elements[0] == 75:
                # Flag is in left orientation
                return Orange_at_left_flag()
        if len(unique_elements)==3:
            if unique_elements[0]==172 :
                # Flag is in correct orientation
               return Correct_flag()
            elif unique_elements[0]==75:
                # Flag is in reversed orientation
                return Reversed_flag()
    # If the image do not have a common edge
    mid_x=int(width/2)
    mid_y=int(height/2)
    #travering vertically
    curr = [row[mid_x] for row in grayscale_image]
    curr1 = []
    for i in curr:
        if i!=0:
            curr1.append(i)
    curr1=get_unique(curr1)
    if len(curr1) == 1:
        #Flag is vertically aligned
        #Traversing horizontally
        curr = [row[mid_y] for row in grayscale_image]
        curr1 = []
        for i in curr:
            if i!=0:
                curr1.append(i)
        curr1=get_unique(curr1)
        if curr1[0] == 172:
            # Flag is in left orientation
            return Orange_at_left_flag()
        else:
            # Flag is in rigth orientation
            return Orange_at_right_flag()
    else :
        #flag is horizontally aligned
        if curr1[0] == 172:
            # Flag is in correct orientation
            return Correct_flag()
        else :
            # Flag is in reversed orientation
            return Reversed_flag()
    # If nothing is returned
    return Correct_flag()