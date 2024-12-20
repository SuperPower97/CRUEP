"""
@author: ZYF
@software: PyCharm
@file: meta_data_process.py
@time: 2024/4/7 16:57
"""

import pandas as pd
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re
import ipdb
import pickle
import json
import os

def text2pkl():
    # 定义文件路径
    txt_file_path = './datasets/Instagram/origin/post_info.txt'  # 替换为您的txt文件路径
    pkl_file_path = './datasets/Instagram/origin/txt2pkl2.pkl'   # 输出的pkl文件路径

    # 读取文本文件到 DataFrame
    df = pd.read_csv(txt_file_path, sep='\t', header=None, names=['id', 'username', 'value', 'user_data_json', 'image_list'], engine='python')

    # 去除重复值
    id_list = df['id'].tolist()
    username_list = df['username'].tolist()
    value_list = df['value'].tolist()
    filename_list = df['user_data_json'].tolist()
    image_list = df['image_list'].tolist()  
    ipdb.set_trace()
    # 将所有列表放入一个大列表中
    all_lists = [id_list, username_list, value_list, filename_list, image_list]
    
    # 保存为 pkl 文件
    with open(pkl_file_path, 'wb') as pkl_file:
        pickle.dump(all_lists, pkl_file)

    print(f"数据已成功保存为 {pkl_file_path}")


def process_meta_data(path):

    file_path = r"./datasets/Instagram/origin/txt2pkl.pkl"

    meta_data = pd.read_pickle(file_path)

    all_pic_name_list = meta_data[4]
    all_user_data_json = meta_data[3]   
    all_user_name_list = meta_data[1]

    image_id_list = []
    text_list = []
    comment_num_list = []
    label_list = []
    user_id_list = []
    taken_timestamp_list = []
    user_name_list = []
    # ipdb.set_trace()
    for i in tqdm(range(len(meta_data))):
        
        if i % 7 != 0:
            
            continue
        
        user_data_json_path = all_user_data_json[i]
        pic_name_list = eval(all_pic_name_list[i])
        # pic_name_list = all_pic_name_list[i]
        user_name = all_user_name_list[i]
        # ipdb.set_trace()
        with open(os.path.join(r"./datasets/Instagram/origin/json", user_data_json_path), 'r',encoding='UTF-8') as json_file:
            
            user_data = json.load(json_file)

            label = user_data["edge_media_preview_like"]["count"]
            
            edge_media_to_caption = user_data["edge_media_to_caption"]["edges"]

            if len(edge_media_to_caption) != 0:
                caption = edge_media_to_caption[0]["node"]["text"]
            else:
                caption = ""

            comment_num = user_data["edge_media_to_comment"]["count"]

            user_id = user_data["owner"]["id"]

            taken_at_timestamp = user_data["taken_at_timestamp"]

        for pic_name in pic_name_list:

            image_id_list.append(pic_name)

            label_list.append(label)

            text_list.append(caption)

            comment_num_list.append(comment_num)

            user_id_list.append(user_id)

            taken_timestamp_list.append(taken_at_timestamp)

            user_name_list.append(user_name)
        # ipdb.set_trace()

    data = {
        "image_id": image_id_list,
        "text": text_list,
        "comment_num": comment_num_list,
        "label": label_list,
        "user_id": user_id_list,
        "taken_timestamp": taken_timestamp_list,
        "user_name": user_name_list
    }

    data_frame = pd.DataFrame(data)
    # ipdb.set_trace()
    data_frame.to_pickle(path)


if __name__ == "__main__":

    path = r"./datasets/Instagram/origin/dataset.pkl"

    # text2pkl()

    dataset = process_meta_data(path)
    print('Process meta data done!')







