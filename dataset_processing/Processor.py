"""TODO
날씨별로 querying하는 함수
전체적인 user-friendly 기능 구축
"""

from PIL import Image
from PIL.ExifTags import TAGS
import os
import json
import shutil
import random
import sys


class Processor():
    def __init__(self):
        pass

    def category_id_edit(self, annot_path):
        '''한 annotation에서 모든 "category_id" 필드 값을 감소시킴
        목적: 로보플로우에서 blackice가 id가 1로 나와서 0으로 바꾸기 위함
        annot_path: 수정하고 싶은 annotation 파일의 경로
        '''
        with open(annot_path, 'r') as f:
            json_data = json.load(f)
            for item in json_data['annotations']:
                item["category_id"] -= 1

            with open(annot_path, 'w') as file:
                json.dump(json_data, file)


    def field_name_change(self, annot_path, class_name):
        '''한 annotation 파일에서 필드명을 바꾸는 함수 - 모든 label을 class_name으로 바꿔버림
        목적: 이륜차 데이터셋에 있는 "침수구간" 이라는 필드명을 클래스 명 중 하나인 "puddle" 로 바꾸기 위함
        annot_path: 수정하려고 하는 annotation 파일의 경로
        class_name: 해당 annotation 파일의 label들을 어떤 class로 바꾸고 싶은지'''
        lst = os.listdir(annot_path)
        for item in lst:
            with open(annot_path + "/" + item, 'r', encoding="UTF8") as f:
                json_data = json.load(f)
                json_data["annotation"][0]["label"] = class_name
                with open(annot_path + "/" + item, 'w') as file:
                    json.dump(json_data, file)


    def random_selection(self, annot_path, new_annot_dir, image_dir, new_image_dir, num_images, file_name):
        '''특정 데이터셋으로부터 num_images 만큼의 이미지를 랜덤하게 선택하고, 이를 annotation과 함께 기존 path에서 new_path로 이동하는 함수
        -file_name은 이동했을 때의 annotation 파일명
        -랜덤 데이터셋 추출-
        annot_path: 기존 annotation 파일 경로
        new_annot_dir: 수정될 annotation 파일의 디렉토리'
        image_dir: 기존 이미지 파일 디렉토리
        new_image_dir: 이미지들이 옮겨질 디렉토리
        num_images: 랜덤하게 선정할 이미지들의 개수
        file_name: 새 annotation 파일의 이름
        '''
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
                new_name = os.path.join(new_image_dir)
                shutil.copy(image_dir + "/" + img, new_name)
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

                
                
            with open(new_annot_dir + "/" + file_name, 'w') as file:
                json.dump(dict, file)


    def sample_selection(self, annot_path, new_annot_dir, image_dir, new_image_dir, file_name):
            '''특정 데이터셋으로부터 특정 이미지에 대한 annotation을 모두 선택하고, 이를 annotation과 함께 기존 path에서 new_path로 이동하는 함수
            -file_name은 이동했을 때의 annotation 파일명
            -랜덤 데이터셋 추출-
            annot_path: 기존 annotation 파일 경로
            new_annot_dir: 수정될 annotation 파일의 디렉토리'
            image_dir: 기존 이미지 파일 디렉토리
            new_image_dir: 이미지들이 옮겨질 디렉토리
            num_images: 랜덤하게 선정할 이미지들의 개수
            file_name: 새 annotation 파일의 이름
            '''
            with open(annot_path , 'r') as f:
                json_data = json.load(f)
                dict = {"info": json_data["info"], "licenses": json_data["licenses"], "categories": json_data["categories"], "images": [], "annotations" : []}
                file_list = os.listdir(image_dir)
                sampled_list = file_list

                img_list_dict = []
                for img in sampled_list:
                    for item in json_data["images"]:
                        if item["file_name"] == img:
                            img_list_dict.append(item)
                    new_name = os.path.join(new_image_dir)
                    shutil.copy(image_dir + "/" + img, new_name)
                sorted_img_list_dict = sorted(img_list_dict, key=lambda d: d['id']) 
                cur_id = 0
                cur_annot_id = 0
                for item in sorted_img_list_dict:
                    dict["images"].append(item)
                    idx = item["id"]
                    dict[-1]["id"] = cur_id
                    for annot in json_data["annotations"]:
                        if annot["image_id"] == idx:
                            dict["annotations"].append(annot)
                            dict["annotations"][-1]["image_id"] = cur_id
                            dict["annotations"][-1]["id"] = cur_annot_id
                            cur_annot_id += 1
                    cur_id += 1
                    
                with open(new_annot_dir + "/" + file_name, 'w') as file:
                    json.dump(dict, file)


    def image_resize(self, image_dir, x, y):
        '''주어진 image_dir 안에 있는 모든 image들을 (x, y) dimension의 이미지로 변환해주는 함수
        image_dir: 이미지 파일들이 위치한 경로
        x: 설정하고자 하는 사진 가로 길이
        y: 설정하고자 하는 사진 세로 길이
        '''
        
        lst = os.listdir(image_dir)
        for img in lst:
            image = Image.open(image_dir + "/" + img)
            new_size = (x, y)
            resized_image = image.resize(new_size)
            resized_image.save(image_dir + "/" + img)


    def dataset_incorporation_bicycle(self, annot_dir, new_annot_dir, file_name, image_dir, new_image_dir):
        '''annot_dir에 있는 모든 annotation file들을 new_annot_dir에 있는 file_name을 가지는 하나의 annotation으로 이전하는 함수
        image_dir에 있는 이미지들을 모두 new_image_dir로 이전하는 역할을 하기도 함
        추가적으로 이륜차 데이터셋이 원 데이터셋의 COCO 형식을 지키기 위해서 필요한 변화를 만들어 줌
        annot_dir: 수정해야 하는 annotation 파일들이 위치한 디렉토리
        new_annot_dir: annotation 파일들이 통합된 annotation이 위치할 디렉토리
        file_name: 새 어노테이션 파일의 이름
        image_dir: 옮겨야 하는 이미지들이 있는 디렉토리
        new_image_dir: image_dir에 있던 이미지들이 옮겨질 디렉토리'''
        cur_id = 0
        annotation_id = 0
        dict = {"categories": [{"id":0,"name":"blackice","supercategory":"none"},{"id":1, "name":"blackice", "supercategory":"blackice"}], "images": [], "annotations" : []}
        for annot in os.listdir(annot_dir):
            with open (annot_dir + "/" + annot, 'r') as f:
                json_data = json.load(f)
                json_data["images"]["id"] = cur_id
                dict["images"].append(json_data["images"])
                for ann in json_data["annotation"]:
                    ann["image_id"] = cur_id
                    ann["id"] = ann.pop("meta_id")
                    ann["category_id"] = ann.pop("label")
                    ann["category_id"] = 1
                    ann["id"] = annotation_id
                    annotation_id += 1
                    dict["annotations"].append(ann)
            cur_id += 1
            new_image_dir = os.path.join(new_image_dir)
            shutil.copy(image_dir + "/" + annot[:-4] + "png", new_image_dir)

        with open(new_annot_dir + "/" + file_name, 'w') as file:
            json.dump(dict, file)


    def indoor_dataset_naming(self, image_dir, annot_path):
        '''실내 데이터셋 이름 바꾸는 함수
        image_dir: 이미지가 위치한 디렉토리
        annot_path: 어노테이션 파일 경로'''
        img_list = os.listdir(image_dir)
        with open(annot_path, 'r') as f:
            json_data = json.load(annot_path)
            num = 1
            for img in img_list:
                image = Image.open(image_dir + "/" + img)
                info = image.getexif()
                taglabel = {}
                for tag, value in info.items():
                    decoded = TAGS.get(tag, tag)
                    taglabel[decoded] = value
                num += 1
                image.close()
                name = image_dir + "/" + "OD_CL_" + taglabel["DateTime"].replace(' ', 'T').replace(':', '') + "_" + "0" * (4 - len(str(num))) + str(num) + ".jpg"
                for item in json_data["images"]:
                    if item["file_name"] == img:
                        item["file_name"] = name
                os.rename(image_dir + "/" + img, image_dir + "/" + "OD_CL_" + taglabel["DateTime"].replace(' ', 'T').replace(':', '') + "_" + "0" * (4 - len(str(num))) + str(num) + ".jpg")



    def outdoor_dataset_naming(self):

        pass


    def bicycle_dataset_naming(self, image_dir, new_image_dir, annot_dir, new_annot_dir):
        '''이륜차 데이터셋 이름 바꾸는 함수
        image_dir: 원래 이미지가 위치해 있는 디렉토리
        new_image_dir: 이미지가 옮겨질 디렉토리
        annot_dir: 원래 annotation이 위치해 있는 디렉토리
        new_annot_dir: annotation이 옮겨질 디렉토리'''
        file_list = os.listdir(image_dir)
        dict = {}
        
        for f in file_list:
            prefix = ""
            datetime = ""
            with open(annot_dir + "/" + f[:-3] + "json", 'r', encoding = "UTF8") as file:
                json_data = json.load(file)
                
                if json_data["images"]["collect_weather"] == "맑음":
                    prefix = "SN"  
                elif json_data["images"]["collect_weather"] == "흐림":
                    prefix = "CL"
                elif json_data["images"]["collect_weather"] == "우천":
                    prefix = "RN"
                else:
                    continue

                try:
                    datetime = json_data["images"]["collect_date"].replace("-", "") + "T" + json_data["images"]["collect_time"].replace(":", "")
                    if datetime in dict:
                        dict[datetime] += 1
                    else:
                        dict[datetime] = 1
                except:
                    continue

                file_name = "/" + "O_" + prefix + "_" + datetime + "_w" + "0" * (5 - len(str(dict[datetime]))) + str(dict[datetime]) + ".png"
                annot_name = "/" + f[:-3] + "json", annot_dir + "/" + "O_" + prefix + "_" + datetime + "_w" + "0" * (5 - len(str(dict[datetime]))) + str(dict[datetime]) + ".json"
                shutil.copy(image_dir + f, new_image_dir + file_name)
                shutil.copy(annot_dir + "/" + f[:-3] + "json", new_annot_dir + annot_name)


    def bicycle_dataset_moving(self, image_dir, annot_dir):
        prefix = os.getcwd()
        img_list = os.listdir(image_dir)
        annot_list = os.listdir(annot_dir)
        


    def bicycle_dataset_querying(self, image_dir, new_image_dir, annot_dir, new_annot_dir):
        """
        목표
        이미지, 라벨링 파일 원하는 개수 가져온 후 학습 하기 전 폴더에 만들어 놓기

        외부데이터
        O_SN_20210825T152815_w0009.png
        외부_날씨_날짜T시간_
        """

        # 날씨에 따라 랜덤으로 maximum N장 가져오기
        # 시간대에 따라 랜덤으로 maximum N장 가져오기

        """
        쿼리 날리는 방법 설명
        """
        how_to_work = """
        <쿼리문 컨벤션>
        {조건 선택}_{조건 범위}_{장수}

        <조건 선택>
        날씨 조건 : 0
        시간 조건 : 1
        -----
        <조건 범위>
        배열을 문자열로 변경한 상태로 넘겨줘야 함.
        CL, SN 날씨를 고르고 싶은 경우 CL SN.
        19~21시 선택하고 싶은 경우 19 20 21.
        ----
        <장수>
        정수 N.

        <예시 쿼리문>
        19, 20, 21시의 데이터 300장 랜덤으로 추출
        1_19 20 21_300

        CL 날씨의 데이터 300장 랜덤으로 추출
        0_CL_300
        """    
        WEATHERLIST = ['CL','SN','RN']
        """
        초기화함수 - img, label 개수 맞는지 체크 후 source dir 경로와 파일 list 리턴.
        """
    def initialize(img_s="원천데이터_230222_add/TS_Bounding Box_27.침수구간", lable_s="라벨링테이터_230222_add/TL_Bounding Box_27.침수구간"):
        # 현재 경로 prefix
        PREFIX = os.getcwd()
        
        img_source_dir = os.path.join(PREFIX, img_s)
        label_source_dir = os.path.join(PREFIX, lable_s)

        img_list = os.listdir(img_source_dir)
        label_list = os.listdir(label_source_dir)

        print(f"img 파일 개수 : {len(os.listdir(img_source_dir))}")
        print(f"label 파일 개수 : {len(os.listdir(label_source_dir))}")

        assert len(img_list) == len(label_list), f"img : {len(img_list)}, label : {len(label_list)} 로 서로 쌍이 맞지 않습니다."

        return img_source_dir, img_list, label_source_dir, label_list

    def check_query(query_condition):
        assert len(query_condition.split('_')) == 3, "쿼리문이 잘못되었습니다."

        condition_type, condition_list, condition_N = query_condition.split('_')

        # type 체크
        assert int(condition_type) == 0 or int(condition_type) == 1, "쿼리문의 condition_type이 잘못되었습니다."

        # 조건 체크
        condition_list = condition_list.split(' ')
        if condition_type==0:
            for c in condition_list:
                assert c in WEATHERLIST, "날씨 조건이 잘못되었습니다."

        # 장수 체크
        condition_N = int(condition_N)

        if int(condition_type) == 0:
            print(f"날씨, 조건 : {condition_list}, 장수 : {condition_N}장")
        elif int(condition_type) == 1:
            print(f"시간, 조건 : {condition_list}, 장수 : {condition_N}장")

        if input("올바르지 않을 경우, n 혹은 N 누르면 종료입니다.").upper()== "N":
            sys.exit()
        
        return int(condition_type), condition_list, int(condition_N)


    def query_weather(condition_list,condition_N):
        print("weather query 시작")
        
    def query_time(condition_list,condition_N):
        print("time query 시작")
        

    # 각 조건에 따른 파일 결과를 미리 보여줌
    def set_query_condition():
        print(how_to_work)
        query_condition = input()
        condition_type, condition_list, condition_N =check_query(query_condition)

        # 날씨 조건 먼저 테스트
        if condition_type == 0:
            query_weather(condition_list,condition_N)
        else:
            query_time(condition_list,condition_N)


    if __name__ == "__main__":
        WEATHERLIST = ['CL','SN','RN']
        TARGET_WEATHER = ''
        N = 300

        # 경로 설정
        img_source_dir, img_list, label_source_dir, label_list = initialize()

        # 쿼리할 파일 설정
        set_query_condition()

        # 쿼리할 파일 결과가 마음에 드는 경우에 파일 이동 등 로직 실행
        if input().upper() == "Y":
            # 파일 이동
            print("do it")
            # 결과 파일 로깅 후 종료    
        else:
            print("exit")
            sys.exit()