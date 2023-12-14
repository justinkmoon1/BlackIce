import os
import numpy as np
import cv2
import matplotlib.pyplot as plt



DATASET_PATH = "DHC OD2.v1i.coco/train"

lst = []
for image in os.listdir(DATASET_PATH):
    if image[-1] != 'g':
        continue
    img = cv2.imread(DATASET_PATH + "/" + image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg = np.mean(hsv[:,:,2])
    
    lst.append(avg)

lst = np.array(lst)

plt.hist(lst, bins=30, range=(0,255), color='lightgreen', edgecolor='blue')

plt.xlabel('Brightness')
plt.ylabel('Frequency')
plt.title('Brightness of images based on HSV model')

plt.show()
print(lst.mean())