import cv2
import numpy as np
import matplotlib.pyplot as plt

# use filter2D
image = cv2.imread(r"pics/lena.png", cv2.IMREAD_GRAYSCALE)

kernel1 = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1],
])
image1 = cv2.filter2D(image, -1, kernel1)

kernel2 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],
])
image2 = cv2.filter2D(image, -1, kernel2)

kernel3 = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
])
image3 = cv2.filter2D(image, -1, kernel3)

plt.subplot(232), plt.imshow(image, 'gray'), plt.title('origin')
plt.subplot(234), plt.imshow(image1, 'gray'), plt.title('filter1')
plt.subplot(235), plt.imshow(image2, 'gray'), plt.title('filter2')
plt.subplot(236), plt.imshow(image3, 'gray'), plt.title('filter3')
plt.show()

# use blur interface
image = cv2.imread(r"pics/lena_with_noisy.jpeg", cv2.IMREAD_GRAYSCALE)

blur_image = cv2.blur(image, (5, 5))
gaussian_image = cv2.GaussianBlur(image, (5, 5), 0.8)
median_image = cv2.medianBlur(image, 5)
bilateral_image = cv2.bilateralFilter(image, 5, 5, 5)

plt.subplot(231), plt.imshow(image, 'gray'), plt.title('lena_with_noisy')
plt.subplot(232), plt.imshow(blur_image, 'gray'), plt.title('blur_image')
plt.subplot(233), plt.imshow(gaussian_image, 'gray'), plt.title('gaussian_image')
plt.subplot(234), plt.imshow(median_image, 'gray'), plt.title('median_image')
plt.subplot(235), plt.imshow(bilateral_image, 'gray'), plt.title('bilateral_image')
plt.show()

