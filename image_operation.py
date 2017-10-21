import cv2
import numpy as np
import matplotlib.pyplot as plt

#resize
image = cv2.imread(r'pics/lena.png', cv2.IMREAD_GRAYSCALE)
shrink_image = cv2.resize(image, None, fx=0.5, fy=0.5)
zoom_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
plt.subplot(131), plt.imshow(image, 'gray'), plt.title('origin')
plt.subplot(132), plt.imshow(shrink_image, 'gray'), plt.title('shrink')
plt.subplot(133), plt.imshow(zoom_image, 'gray'), plt.title('zoom')
plt.show()

image = cv2.imread(r'pics/lena.png', cv2.IMREAD_GRAYSCALE)

# translation
translation_M = np.array([[1, 0, 20], [0, 1, 20]], dtype=np.float32)
translation_image = cv2.warpAffine(image, translation_M, image.shape)

# rotation
rows, cols = image.shape
rotation_M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotation_image = cv2.warpAffine(image, rotation_M, image.shape)

# offset
offset_M = np.array([[1, 0.5, 0], [0, 1, 0]], dtype=np.float32)
offset_image = cv2.warpAffine(image, offset_M, image.shape)

plt.subplot(221), plt.imshow(image, 'gray'), plt.title('origin')
plt.subplot(222), plt.imshow(translation_image, 'gray'), plt.title('translation')
plt.subplot(223), plt.imshow(rotation_image, 'gray'), plt.title('rotation')
plt.subplot(224), plt.imshow(offset_image, 'gray'), plt.title('offset')
plt.show()

