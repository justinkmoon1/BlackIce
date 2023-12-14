import os
import numpy as np
import cv2
import matplotlib.pyplot as plt



DATASET_PATH = "/Users/hanseoghui/Downloads/OD2/train"

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

fs_title = 18
plt.title('Brightness of images based on HSV model', fontsize=fs_title)

fs_xylabel = 16
plt.xlabel('Brightness', fontsize=fs_xylabel)
plt.ylabel('Frequency', fontsize=fs_xylabel)

fs_ticks = 14
plt.xticks(fontsize=fs_ticks)
plt.yticks(fontsize=fs_ticks)

plt.savefig('/Users/hanseoghui/Downloads/brightness_OD2.jpg')
#plt.show()
#print(lst.mean())