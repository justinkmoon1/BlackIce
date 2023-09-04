import json
import os
import random
import shutil

ANNOT_PATH_TRAIN = 'BlackIce/datasets/COCO/annotations_real/_annotations.coco.json'
#원래 어노테이션 파일 경로 (train)

ANNOT_PATH_VAL = 'BlackIce/datasets/COCO/annotations_real/_annotations.coco.json'
#원래 어노테이션 파일 경로 (val)

IMAGE_DIR_TRAIN = 'BlackIce/datasets/COCO/train2017_real'
#원래 이미지 폴더 경로 (train)

IMAGE_DIR_VAL = 'BlackIce/datasets/COCO/val2017_real'
#원래 이미지 폴더 경로 (validation)

NEW_ANNOT_DIR = 'BlackIce/datasets/COCO/annotations'
#어노테이션 파일이 이동한 후 위치할 폴더의 경로

NEW_IMAGE_DIR_TRAIN = 'BlackIce/datasets/COCO/train2017'
#이미지 파일이 이동한 후 위치할 폴더의 경로 (train)

NEW_IMAGE_DIR_VAL = 'BlackIce/datasets/COCO/val2017'
#이미지 파일이 이동한 후 위치할 폴더의 경로 (validation)

FILE_NAME_TRAIN = '_annotations.coco.json'
#어노테이션 파일명

FILE_NAME_VAL = '_annotation.coco.val.json'
#어노테이션 파일명

N = 10 #train
M = 2 #valid
# #이미지 몇 장을 선정할 예정인지

# def selection(annot_path, new_annot_path, image_dir, new_image_dir, num_images, file_name):
#     with open(annot_path , 'r') as f:
#         json_data = json.load(f)
#         dict = {"info": json_data["info"], "licenses": json_data["licenses"], "categories": json_data["categories"], "images": [], "annotations" : []}
#         file_list = os.listdir(image_dir)
#         sampled_list = random.sample(file_list, num_images)

#         cur_id = 0
#         cur_annot_id = 0
#         for img in sampled_list:
#             for item in json_data["images"]:
#                 if item["file_name"] == img:
#                     idx = item["id"]
#                     dict["images"].append(item)
#                     dict["images"][-1]["id"] = cur_id
#                     for annot in json_data["annotations"]:
#                         if annot["image_id"] == idx:
#                             dict["annotations"].append(annot)
#                             dict["annotations"][-1]["image_id"] = cur_id
#                             dict["annotations"][-1]["id"] = cur_annot_id
#                             cur_annot_id += 1
#             cur_id += 1

#             new_name = os.path.join(new_image_dir)
#             shutil.copy(image_dir + "/" + img, new_name)   
#         with open(new_annot_path + "/" + file_name, 'w') as file:
#             json.dump(dict, file)

# selection(ANNOT_PATH_TRAIN, NEW_ANNOT_DIR, IMAGE_DIR_TRAIN, NEW_IMAGE_DIR_TRAIN, N, FILE_NAME_TRAIN)
# selection(ANNOT_PATH_VAL, NEW_ANNOT_DIR, IMAGE_DIR_VAL, NEW_IMAGE_DIR_VAL, M, FILE_NAME_VAL)


#어노테이션 파일명

N = 10 #train
M = 2 #valid
#이미지 몇 장을 선정할 예정인지

def selection(annot_path, new_annot_path, image_dir, new_image_dir, num_images, file_name):
    with open(annot_path , 'r') as f:
        json_data = json.load(f)
        dict = {"info": json_data["info"], "licenses": json_data["licenses"], "categories": json_data["categories"], "images": [], "annotations" : []}
        file_list = os.listdir(image_dir)
        sampled_list = random.sample(file_list, num_images)

        img_list_dict = []
        for img in sampled_list:
            for item in json_data["images"]:
                if item["file_name"] == img:
                    img_list_dict.append(item)
        sorted_img_list_dict = sorted(img_list_dict, key=lambda d: d['id']) 
        cur_id = 0
        cur_annot_id = 0
        for item in sorted_img_list_dict:
            dict["images"].append(item)
            idx = item["id"]
            dict["images"][-1]["id"] = cur_id
            for annot in json_data["annotations"]:
                if annot["image_id"] == idx:
                    dict["annotations"].append(annot)
                    dict["annotations"][-1]["image_id"] = cur_id
                    dict["annotations"][-1]["id"] = cur_annot_id
                    cur_annot_id += 1
            cur_id += 1

            new_name = os.path.join(new_image_dir)
            shutil.copy(image_dir + "/" + item["file_name"], new_name)
        with open(new_annot_path + "/" + file_name, 'w') as file:
            json.dump(dict, file)

selection(ANNOT_PATH_TRAIN, NEW_ANNOT_DIR, IMAGE_DIR_TRAIN, NEW_IMAGE_DIR_TRAIN, N, FILE_NAME_TRAIN)
selection(ANNOT_PATH_VAL, NEW_ANNOT_DIR, IMAGE_DIR_VAL, NEW_IMAGE_DIR_VAL, M, FILE_NAME_VAL)