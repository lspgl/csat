import numpy as np
import cv2


def find_skeleton(img):
    # Initialize arrays to do computation in place
    skeleton = np.zeros(img.shape, np.uint8)
    eroded = np.zeros(img.shape, np.uint8)
    temp = np.zeros(img.shape, np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while(True):
        cv2.erode(img, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        cv2.subtract(img, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        img, eroded = eroded, img  # Swap instead of copy

        if cv2.countNonZero(img) == 0:
            return skeleton
