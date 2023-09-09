import cv2 as cv
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import matplotlib.pyplot as plt
import numpy as np

def bounding_box(image):
    segmentor = SelfiSegmentation()
    segmented_image = segmentor.removeBG(image,(0,0,0))
    gray_image = cv.cvtColor(segmented_image, cv.COLOR_BGR2GRAY)
    
    contours, _ = cv.findContours(gray_image.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        M = cv.moments(largest_contour)
        # Calculate the centroid coordinates
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        x, y, w, h = cv.boundingRect(largest_contour)

        shift_percent = 0.1555
        new_center_x = centroid_x
        new_center_y = int(centroid_y - (h * shift_percent))  # Shift upward by 20 percent of the bounding box height

    # Update the center coordinates
    center = (new_center_x-50, new_center_y)   
    major_axis = h-100 
    minor_axis = w
    angle = 90  # Major axis at 90 degrees (vertical)

    rectangle_height = major_axis 
    rectangle_width = int(rectangle_height * 4 / 5) 

    # Calculate the coordinates of the rectangle's top-left and bottom-right corners
    x1 = center[0] - rectangle_width // 2
    y1 = center[1] - rectangle_height // 2
    x2 = center[0] + rectangle_width // 2
    y2 = center[1] + rectangle_height // 2

    # Draw the rectangle
    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=6)
    
    return image

# Load the image
image = cv.imread(image_path)  # Replace with your image file path
result_image = bounding_box(image)
plt.figure(figsize=[22, 22])
plt.subplot(1, 2, 1)
plt.title('original')
plt.axis('off')
plt.imshow(image[:, :, ::-1])
plt.show()

        