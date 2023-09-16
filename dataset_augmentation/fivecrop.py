import albumentations as A
import cv2
import os
import json
import shutil

DATA_PATH = "Test Set Augmentation/train"

NEW_ANNOT_DIR = "Test Set Augmentation/"

FILE_NAME = "annotations_augmented.json"

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
    cur_id = json_data["images"][-1]["id"] + 1
    cur_annot_id = json_data["annotations"][-1]["id"] + 1
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
        transformed2 = transform2(image=image, bboxes=bbox, class_labels=label)
        transformed3 = transform3(image=image, bboxes=bbox, class_labels=label)
        
        bbox0 = transformed['bboxes']
        label0 = transformed['class_labels']

        bbox1 = transformed1['bboxes']
        label1 = transformed1['class_labels']
        
        bbox2 = transformed2['bboxes']
        label2 = transformed2['class_labels']

        bbox3 = transformed3['bboxes']
        label3 = transformed3['class_labels']

        cv2.imwrite(DATA_PATH + "/" + name[:-4] + "transformed" + "0" + ".jpg", transformed['image'])
        cv2.imwrite(DATA_PATH + "/" + name[:-4] + "transformed" + "1" + ".jpg", transformed1['image'])
        cv2.imwrite(DATA_PATH + "/" + name[:-4] + "transformed" + "2" + ".jpg", transformed2['image'])
        cv2.imwrite(DATA_PATH + "/" + name[:-4] + "transformed" + "3" + ".jpg", transformed3['image'])
        for i in range(4):
            file_name = name[:-4] + "transformed" + str(i) + ".jpg"
            im = cv2.imread(DATA_PATH + "/" + file_name)
            image_dict = {"id": cur_id, "license" : 1, "file_name": name, "height": im.shape[0], "width": im.shape[1]}
            json_data["images"].append(image_dict)
            if i == 0:
                for j in range(len(bbox0)):
                    annot_dict = {"id": cur_annot_id, "image_id": cur_id, "category_id": label0[j], "bbox": bbox0[j], "area": bbox0[j][2] * bbox0[j][3], "segmentation": [], "iscrowd":0}
                    json_data["annotations"].append(annot_dict)
                    cur_annot_id += 1
            if i == 1:
                for j in range(len(bbox1)):
                    annot_dict = {"id": cur_annot_id, "image_id": cur_id, "category_id": label1[j], "bbox": bbox1[j], "area": bbox1[j][2] * bbox1[j][3], "segmentation": [], "iscrowd":0}
                    json_data["annotations"].append(annot_dict)
                    cur_annot_id += 1
            
            if i == 2:
                for j in range(len(bbox2)):
                    annot_dict = {"id": cur_annot_id, "image_id": cur_id, "category_id": label2[j], "bbox": bbox2[j], "area": bbox2[j][2] * bbox2[j][3], "segmentation": [], "iscrowd":0}
                    json_data["annotations"].append(annot_dict)

                    cur_annot_id += 1

            if i == 3:
                for j in range(len(bbox3)):
                    annot_dict = {"id": cur_annot_id, "image_id": cur_id, "category_id": label3[j], "bbox": bbox3[j], "area": bbox3[j][2] * bbox3[j][3], "segmentation": [], "iscrowd":0}
                    json_data["annotations"].append(annot_dict)

                    cur_annot_id += 1

            cur_id += 1
        
    with open(NEW_ANNOT_DIR + FILE_NAME, 'w') as file:
        json.dump(json_data, file)



            
    