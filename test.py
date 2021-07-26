# test.py

from python_cpp_module import test_func, test_image_processing, npy_array_test, test_image_processing_OCV
import numpy as np
import cv2

image_file = './resources/car_1.bmp'

# Simple addition function test
print(test_func(1,3))    # Anser should be 4.

# Numpy array parameter extraction test
a=np.zeros((480,640,3), dtype=np.uint8)
a[0,0,0:3] = [5,6,7]
npy_array_test(a)

# Simple image processing test (inverse color)
img = cv2.imread(image_file)
cv2.imshow('input', img)
img=test_image_processing(img)
cv2.imshow('inversed', img)

# Simple image processing test with OpenCV (Canny edge detection)
img = cv2.imread(image_file)
img = test_image_processing_OCV(img, 100, 200)
cv2.imshow('Canny', img)
cv2.waitKey(0)
