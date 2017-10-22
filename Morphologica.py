import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r'pics/circle.png')
kernel = np.ones((10, 10), np.uint8)
erosion = cv2.erode(image, kernel, iterations=1)

plt.subplot(121), plt.imshow(image), plt.title('Origin')
plt.subplot(122), plt.imshow(erosion), plt.title('Erode')
plt.show()

image = cv2.imread(r'pics/circle.png')
kernel = np.ones((5, 5), np.uint8)
dilate = cv2.dilate(image, kernel, iterations=1)

plt.subplot(121), plt.imshow(image), plt.title('Origin')
plt.subplot(122), plt.imshow(dilate), plt.title('Dilate')
plt.show()

image = cv2.imread(r'pics/black_circle.png')
kernel = np.ones((5, 5), np.uint8)
open_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

plt.subplot(121), plt.imshow(image), plt.title('Origin')
plt.subplot(122), plt.imshow(open_img), plt.title('Open')
plt.show()

image = cv2.imread(r'pics/white_circle.png')
kernel = np.ones((5, 5), np.uint8)
close_img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

plt.subplot(121), plt.imshow(image), plt.title('Origin')
plt.subplot(122), plt.imshow(close_img), plt.title('Close')
plt.show()

image = cv2.imread(r'pics/circle.png')
kernel = np.ones((5, 5), np.uint8)
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

plt.subplot(121), plt.imshow(image), plt.title('Origin')
plt.subplot(122), plt.imshow(gradient), plt.title('Gradient')
plt.show()
