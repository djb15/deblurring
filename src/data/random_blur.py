import cv2
import os
import numpy as np

# Path for the directory containing raw images
BASE = '../../data/raw/'
OUT = '../../data/processed/'

raw_photos = os.listdir(BASE)
count = 0

for photo in raw_photos:
    count += 1
    path = BASE + photo
    img = cv2.imread(path)

    # Randomise size of matrix for motion blur
    size = 10

    kernel_motion_blur = np.zeros((size, size))
    # Vary direction of values inside matrix for different blur direction
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    output = cv2.filter2D(img, -1, kernel_motion_blur)
    cv2.imwrite(OUT + str(count) + '.jpg', output)
