import cv2
import os
import numpy as np
import random


def random_kernel_size(image_width):
    # max blur is 12.5% of image width
    max_size = np.ceil(0.25 * image_width)
    size = int(max_size * random.uniform(0.3, 1.0))
    # nearest odd number
    size = size - 1 if size % 2 is 0 else size
    return size


def rotate_bound(image, angle, original_dims=None):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # if original_dims is set, crop to original size
    if not original_dims:
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform rotation
        return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_REPLICATE)
    else:
        # translate image to top-left and then rotate back
        M[0, 2] -= (w - original_dims[0]) / 2
        M[1, 2] -= (h - original_dims[1]) / 2
        return cv2.warpAffine(image, M, original_dims)
    


if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    input_img_dir = os.path.join(project_dir, "data", "raw", "pre-blur")
    output_dir = os.path.join(project_dir, "data", "processed")

    raw_photos = os.listdir(input_img_dir)
    count = 0
    for photo in raw_photos[:50]:
        count += 1
        path = os.path.join(input_img_dir, photo)
        img = cv2.imread(path)

        # Randomise size of matrix for motion blur
        height, width = img.shape[:2]
        size = random_kernel_size(width)
        
        # random angle between 0 and 180
        angle = random.uniform(0, 180)
        # rotate image by random angle
        img = rotate_bound(img, angle)

        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int(size/2), int(size/2):] = np.ones(int((size+1)/2))
        kernel_motion_blur = kernel_motion_blur / int((size+1)/2)
        # perform motion blur
        output_img = cv2.filter2D(img, -1, kernel_motion_blur, borderType=cv2.BORDER_REPLICATE)
        # unrotate image, cropping to original dimensions
        output_img = rotate_bound(output_img, -angle, (width, height))

        output_filepath = os.path.join(output_dir, "{}.jpg".format(count))
        cv2.imwrite(output_filepath, output_img)
