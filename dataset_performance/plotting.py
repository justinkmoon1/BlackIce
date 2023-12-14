import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

DATASET_PATH = "DHC OD2.v1i.coco/train"
ANNOT_PATH = "DHC OD2.v1i.coco/train/_annotations.coco.json"

last, axs = plt.subplots(1, 2, tight_layout=True)

lst = []
with open(ANNOT_PATH, 'r') as f:
    json_data = json.load(f)
    for image in os.listdir(DATASET_PATH):
        if image[-1] != 'g':
            continue
        dims = 0 
        area = 0
        id = -1
        for annot in json_data["images"]:
            if annot["file_name"] == image:
                id = annot["id"]
                dims = annot["height"] * annot["width"]
        for annot in json_data["annotations"]:
            if annot["image_id"] == id:
                area += annot["area"]
        lst.append((area/dims) * 100)


lst = np.array(lst)



axs[0].hist(lst, bins=30, range=(0,100), color='lightgreen', edgecolor='blue')


axs[0].set_xlabel('Proportion of instance pixel area (%)', fontsize=20)
axs[0].set_ylabel('Frequency', fontsize=20)
axs[0].set_title('Proportion of instance pixel area', fontsize=20)






lst = []
for image in os.listdir(DATASET_PATH):
    if image[-1] != 'g':
        continue
    img = cv2.imread(DATASET_PATH + "/" + image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg = np.mean(hsv[:,:,2])
    
    lst.append(avg)

lst = np.array(lst)

axs[1].hist(lst, bins=30, range=(0,255), color='lightgreen', edgecolor='blue')

axs[1].set_xlabel('Brightness', fontsize=20)
axs[1].set_ylabel('Frequency', fontsize=20)
axs[1].set_title('Brightness of images based on HSV model', fontsize=20)
        


plt.show()
