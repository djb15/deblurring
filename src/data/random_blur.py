import cv2
import os
import numpy as np
import random

# Path for the directory containing raw images
BASE = '../../data/raw/'
OUT = '../../data/processed/'

raw_photos = os.listdir(BASE)
count = 0

def random_kernel_size(image_width):
    # max blur is 12.5% of image width
    max_size = np.ceil(0.25 * image_width)
    size = int(max_size * random.uniform(0.3,1.0))
    # nearest odd number
    size = size - 1 if size % 2 is 0 else size
    return size

for photo in raw_photos[:50]:
    count += 1
    path = BASE + photo
    img = cv2.imread(path)

    # Randomise size of matrix for motion blur
    height, width = img.shape[:2]
    size = random_kernel_size(width)

    kernel_motion_blur = np.zeros((size, size))
    # Vary direction of values inside matrix for different blur direction
    kernel_motion_blur[int(size/2), int(size/2):] = np.ones(int((size+1)/2))
    kernel_motion_blur = kernel_motion_blur / int((size+1)/2)

    output = cv2.filter2D(img, -1, kernel_motion_blur)

    cv2.imwrite(OUT + str(count) + '.jpg', output)
