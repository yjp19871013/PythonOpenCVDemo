import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r'pics/lena.png', cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

equ_image = cv2.equalizeHist(image)
equ_hist = cv2.calcHist([equ_image], [0], None, [256], [0, 256])

clahe = cv2.createCLAHE(clipLimit=3.0)
clahe_image = clahe.apply(image)
clahe_hist = cv2.calcHist([clahe_image], [0], None, [256], [0, 256])

plt.subplot(231), plt.imshow(image, 'gray'), plt.title('Origin')
plt.subplot(232), plt.imshow(equ_image, 'gray'), plt.title('Equalize')
plt.subplot(233), plt.imshow(clahe_image, 'gray'), plt.title('Clahe')

plt.subplot(234)
plt.hist(hist.flatten(), 256, [0, 256], color='b')
plt.title('Origin Hist')

plt.subplot(235)
plt.hist(equ_hist.flatten(), 256, [0, 256], color='b')
plt.title('Equalize Hist')

plt.subplot(236)
plt.hist(clahe_hist.flatten(), 256, [0, 256], color='b')
plt.title('Equalize Hist')
plt.show()


