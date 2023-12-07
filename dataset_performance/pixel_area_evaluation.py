import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

DATASET_PATH = "datasets/test2017"
ANNOT_PATH = "datasets/annotations/_annotations.coco.json"

lst = []
with open(ANNOT_PATH, 'r') as f:
    json_data = json.load(f)
    for image in os.listdir(DATASET_PATH):
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

plt.hist(lst, bins=30, range=(0,100), color='lightgreen', edgecolor='blue')

plt.xlabel('Proportion of instance pixel area (%)')
plt.ylabel('Frequency')
plt.title('Proportion of instance pixel area')

plt.show()

        


    
