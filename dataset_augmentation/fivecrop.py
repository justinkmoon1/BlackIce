import albumentations as A
import cv2
import os
import json
import shutil

DATA_PATH = "Test Set Augmentation/train"

ANNOT_PATH = "Test Set Augmentation\_annotations.coco.json"

transform = A.Compose([
    A.RandomCropFromBorders(crop_bottom = 1, 
                            always_apply=True)], 
                            bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
transform1 = A.Compose([
    A.RandomCropFromBorders(crop_top = 1, 
                            always_apply=True)], 
                            bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
transform2 = A.Compose([
    A.RandomCropFromBorders(crop_left = 1, 
                            always_apply=True)], 
                            bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
transform3 = A.Compose([
    A.RandomCropFromBorders(crop_right = 1, 
                            always_apply=True)], 
                            bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

with open(ANNOT_PATH, 'r') as f:
    json_data = json.load(f)
    for img in json_data["images"]:
        name = img["file_name"]
        idx =  img["id"]
        image = cv2.imread(DATA_PATH + "/" + name)
        
        bbox = []
        label = []
        for annot in json_data["annotations"]:
             if annot["image_id"] == idx:
                lst = annot["bbox"]
                bbox.append(lst)
                label.append(annot["category_id"])

        transformed = transform(image=image, bboxes=bbox, class_labels=label)
        transformed1 = transform1(image=image, bboxes=bbox, class_labels=label)
        transformed2 = transform1(image=image, bboxes=bbox, class_labels=label)
        transformed3 = transform1(image=image, bboxes=bbox, class_labels=label)
        
        cv2.imwrite(DATA_PATH + "/" + name[:-4] + "transformed" + "0" + ".jpg", transformed['image'])
        cv2.imwrite(DATA_PATH + "/" + name[:-4] + "transformed" + "1" + ".jpg", transformed1['image'])
        cv2.imwrite(DATA_PATH + "/" + name[:-4] + "transformed" + "2" + ".jpg", transformed2['image'])
        cv2.imwrite(DATA_PATH + "/" + name[:-4] + "transformed" + "3" + ".jpg", transformed3['image'])



            
    