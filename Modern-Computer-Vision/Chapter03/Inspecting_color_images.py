import os
import cv2, matplotlib.pyplot as plt
print(os.getcwd())

img = cv2.imread('../../data/image/Hemanvi.jpeg')
img = img[50:250,40:240,:]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

plt.imshow(img)
plt.show()
print('img.shape:', img.shape)

