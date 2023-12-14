import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

DATASET_PATH = "/Users/hanseoghui/Downloads/OD2/train"
ANNOT_PATH = "/Users/hanseoghui/Downloads/OD2/train/_annotations.coco.json"

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

print(lst.mean())
plt.hist(lst, bins=30, range=(0,100), color='lightgreen', edgecolor='blue')

fs_title = 18
plt.title('Proportion of instance pixel area', fontsize=fs_title)

fs_xylabel = 16
plt.xlabel('Proportion of instance pixel area (%)', fontsize=fs_xylabel)
plt.ylabel('Frequency', fontsize=fs_xylabel)

fs_ticks = 14
plt.xticks(fontsize=fs_ticks)
plt.yticks(fontsize=fs_ticks)

plt.savefig('/Users/hanseoghui/Downloads/pixelarea_OD2.jpg')
#plt.show()