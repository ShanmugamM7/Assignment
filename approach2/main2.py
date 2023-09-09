import os
import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

def bounding_box(image):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segment = mp_selfie_segmentation.SelfieSegmentation()

    result = segment.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    mask = result.segmentation_mask > 0.2

    # Finding contours
    contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv.contourArea)  
        x, y, w, h = cv.boundingRect(largest_contour)
        y -= 150
        mid_x = x + w // 2
        mid_y = y + h // 2 
        proportion = 0.80
        if w / h > proportion:
            new_h = int(w / proportion)
            y = mid_y - new_h // 2
            h = new_h
        else:
            new_w = int(h * proportion)
            x = mid_x - new_w // 2
            w = new_w
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 10)

        return image

# Loading images
image_path = 'IMG_0842.JPG'    # give the image path here
image = cv.imread(image_path)
result_image = bounding_box(image)

plt.figure(figsize=[22, 22])
plt.subplot(1, 2, 1)
plt.title('original')
plt.axis('off')
plt.imshow(image[:, :, ::-1])
plt.show()