import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def bounding_box(image):
    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur and thresholding
    blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)
    _, thresholded_image = cv.threshold(blurred_image, 127, 255, cv.THRESH_BINARY)

    # Find contours
    contours, _ = cv.findContours(thresholded_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)  # Largest contour
        M = cv.moments(largest_contour)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        x, y, w, h = cv.boundingRect(largest_contour)

    shift_percent = 0.6
    new_center_x = centroid_x
    new_center_y = int(centroid_y + (h * shift_percent))  
    center = (new_center_x, new_center_y)
    major_axis = h
    minor_axis = w

    rectangle_height = major_axis  
    rectangle_width = int(rectangle_height * 4 / 5)
    x1 = center[0] - rectangle_width // 2
    y1 = center[1] - rectangle_height // 2
    x2 = center[0] + rectangle_width // 2
    y2 = center[1] + rectangle_height // 2

    # Draw the rectangle
    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=6)
    return image

# Load the image
image = cv.imread('IMG_0852.JPG')  # Replace with your image file path
result_image = bounding_box(image)
plt.figure(figsize=[22, 22])
plt.subplot(1, 2, 1)
plt.title('original')
plt.axis('off')
plt.imshow(image[:, :, ::-1])
plt.show()


        
