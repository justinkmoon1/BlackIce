import os
import numpy as np
import cv2
import matplotlib.pyplot as plt



DATASET_PATH = "datasets/test2017"

lst = []
for image in os.listdir(DATASET_PATH):
    img = cv2.imread(DATASET_PATH + "/" + image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg = np.mean(hsv[:,:,2])
    per = (avg/255) * 100
    lst.append(per)

lst = np.array(lst)

plt.hist(lst, bins=30, range=(0,100), color='lightgreen', edgecolor='blue')

plt.xlabel('Brightness (%)')
plt.ylabel('Frequency')
plt.title('Brightness of images based on HSV')

plt.show()
print(lst.mean())