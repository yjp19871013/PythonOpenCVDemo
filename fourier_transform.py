import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r'pics/lena.png', cv2.IMREAD_GRAYSCALE)

dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

image_back = np.fft.ifftshift(dft_shift)
image_back = cv2.idft(image_back)
image_back = cv2.magnitude(image_back[:, :, 0], image_back[:, :, 1])

rows, cols = image.shape
center_x, center_y = int(rows/2), int(cols/2)
dft_shift[center_x-30:center_x+30, center_y-30:center_y+30] = 0
filter_back = np.fft.ifftshift(dft_shift)
filter_back = cv2.idft(filter_back)
filter_back = cv2.magnitude(filter_back[:, :, 0], filter_back[:, :, 1])

plt.subplot(221), plt.imshow(image, 'gray'), plt.title('Origin')
plt.subplot(222), plt.imshow(magnitude_spectrum, 'gray'), plt.title('Fourier TransForm')
plt.subplot(223), plt.imshow(image_back, 'gray'), plt.title('Origin Back')
plt.subplot(224), plt.imshow(filter_back, 'gray'), plt.title('Transform Back')
plt.show()


