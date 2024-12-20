import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import ipdb
import math


def create_retrieval_pool(train_path, valid_path, retrieval_pool_path):

    train_data = pd.read_pickle(train_path)
    valid_data = pd.read_pickle(valid_path)

    retrieval_pool = pd.concat([train_data, valid_data], axis=0)
    retrieval_pool.reset_index(drop=True, inplace=True)

    retrieval_pool.to_pickle(retrieval_pool_path)

    return retrieval_pool


def calculate_similarity(all_retrieval_result, N):
    # 计算每个特征值出现的次数
    n_values = all_retrieval_result.sum(axis=0)

    def f_similarity(n):
        # 计算相似度
        return np.log((N - n + 0.5) / (n + 0.5))

    # 计算相似度
    similarity = np.dot(all_retrieval_result, f_similarity(n_values))
    return similarity


def retrieval_data(retrieval_num, data_path, retrieval_pool_path):
    # 读取数据集和待检索数据
    dataset = pd.read_pickle(retrieval_pool_path)
    data = pd.read_pickle(data_path)

    # 获取所有特征列
    all_features = ["comment_num",'user_id', 'taken_timestamp']
    
    # 转换为 Numpy 数组，以便进行高效的向量化操作
    dataset_array = dataset[all_features].values
    data_array = data[all_features].values

    # 计算数据集大小
    N = len(dataset)

    # 存储检索结果的列表
    retrieved_item_id_list = []
    retrieved_item_similarity_list = []
    retrieved_label_list = []

    # 遍历待检索数据
    for i in tqdm(range(len(data))):
        # 获取查询特征向量
        query_features = data_array[i]

        # 计算相似度
        similarities = calculate_similarity((dataset_array == query_features).astype(int), N)

        # 将自身相似度置为0
        similarities[i] = 0

        # 获取相似度排序后的索引
        retrieval_indices = np.argsort(similarities)[::-1][:retrieval_num]
        retrieved_items = dataset.iloc[retrieval_indices]

        # 提取检索结果的相关信息，并存储到列表中
        retrieved_item_id_list.append(retrieved_items['image_id'].tolist())
        retrieved_item_similarity_list.append(similarities[retrieval_indices].tolist())
        retrieved_label_list.append(retrieved_items['label'].tolist())

    # 将检索结果存储到待检索数据中
    data['retrieved_item_id'] = retrieved_item_id_list
    data['retrieved_item_similarity'] = retrieved_item_similarity_list
    data['retrieved_label'] = retrieved_label_list

    # 存储结果到文件
    data.to_pickle(data_path)


# def stack_retrieved_feature(train_path, valid_path, test_path):

#     df_train = pd.read_pickle(train_path)

#     df_test = pd.read_pickle(test_path)

#     df_valid = pd.read_pickle(valid_path)

#     df_database = pd.concat([df_train, df_test, df_valid], axis=0)

#     df_database.reset_index(drop=True, inplace=True)

#     retrieved_visual_feature_embedding_cls_list = []

#     retrieved_visual_feature_embedding_mean_list = []

#     retrieved_textual_feature_embedding_list = []

#     retrieve_label_list = []

#     for i in tqdm(range(len(df_train))):

#         id_list = df_train['retrieved_item_id'][i]

#         current_retrieved_visual_feature_embedding_cls_list = []

#         current_retrieved_visual_feature_embedding_mean_list = []

#         current_retrieved_textual_feature_embedding_list = []

#         current_retrieved_label_list = []

#         for j in range(len(id_list)):
#             item_id = id_list[j]

#             index = df_database[df_database['image_id'] == item_id].index[0]

#             current_retrieved_visual_feature_embedding_cls_list.append(
#                 df_database['cls_vec'][index])

#             current_retrieved_visual_feature_embedding_mean_list.append(
#                 df_database['mean_pooling_vec'][index])

#             current_retrieved_textual_feature_embedding_list.append(df_database['merged_text_vec'][index])

#             current_retrieved_label_list.append(df_database['label'][index])

#         retrieved_visual_feature_embedding_cls_list.append(current_retrieved_visual_feature_embedding_cls_list)

#         retrieved_visual_feature_embedding_mean_list.append(current_retrieved_visual_feature_embedding_mean_list)

#         retrieved_textual_feature_embedding_list.append(current_retrieved_textual_feature_embedding_list)

#         retrieve_label_list.append(current_retrieved_label_list)

#     df_train['retrieved_visual_feature_embedding_cls'] = retrieved_visual_feature_embedding_cls_list

#     df_train['retrieved_visual_feature_embedding_mean'] = retrieved_visual_feature_embedding_mean_list

#     df_train['retrieved_textual_feature_embedding'] = retrieved_textual_feature_embedding_list

#     df_train['retrieved_label_list'] = retrieve_label_list

#     df_train.to_pickle(train_path)

#     retrieved_visual_feature_embedding_cls_list = []

#     retrieved_visual_feature_embedding_mean_list = []

#     retrieved_textual_feature_embedding_list = []

#     retrieve_label_list = []

#     for i in tqdm(range(len(df_test))):

#         id_list = df_test['retrieved_item_id'][i]

#         current_retrieved_visual_feature_embedding_cls_list = []

#         current_retrieved_visual_feature_embedding_mean_list = []

#         current_retrieved_textual_feature_embedding_list = []

#         current_retrieved_label_list = []

#         for j in range(len(id_list)):
#             item_id = id_list[j]

#             index = df_database[df_database['image_id'] == item_id].index[0]

#             current_retrieved_visual_feature_embedding_cls_list.append(
#                 df_database['cls_vec'][index])

#             current_retrieved_visual_feature_embedding_mean_list.append(
#                 df_database['mean_pooling_vec'][index])

#             current_retrieved_textual_feature_embedding_list.append(df_database['merged_text_vec'][index])

#             current_retrieved_label_list.append(df_database['label'][index])

#         retrieved_visual_feature_embedding_cls_list.append(current_retrieved_visual_feature_embedding_cls_list)

#         retrieved_visual_feature_embedding_mean_list.append(current_retrieved_visual_feature_embedding_mean_list)

#         retrieved_textual_feature_embedding_list.append(current_retrieved_textual_feature_embedding_list)

#         retrieve_label_list.append(current_retrieved_label_list)

#     df_test['retrieved_visual_feature_embedding_cls'] = retrieved_visual_feature_embedding_cls_list

#     df_test['retrieved_visual_feature_embedding_mean'] = retrieved_visual_feature_embedding_mean_list

#     df_test['retrieved_textual_feature_embedding'] = retrieved_textual_feature_embedding_list

#     df_test['retrieved_label_list'] = retrieve_label_list

#     df_test.to_pickle(test_path)

#     retrieved_visual_feature_embedding_cls_list = []

#     retrieved_visual_feature_embedding_mean_list = []

#     retrieved_textual_feature_embedding_list = []

#     retrieve_label_list = []

#     for i in tqdm(range(len(df_valid))):

#         id_list = df_valid['retrieved_item_id'][i]

#         current_retrieved_visual_feature_embedding_cls_list = []

#         current_retrieved_visual_feature_embedding_mean_list = []

#         current_retrieved_textual_feature_embedding_list = []

#         current_retrieved_label_list = []

#         for j in range(len(id_list)):
#             item_id = id_list[j]

#             index = df_database[df_database['image_id'] == item_id].index[0]

#             current_retrieved_visual_feature_embedding_cls_list.append(
#                 df_database['cls_vec'][index])

#             current_retrieved_visual_feature_embedding_mean_list.append(
#                 df_database['mean_pooling_vec'][index])

#             current_retrieved_textual_feature_embedding_list.append(df_database['merged_text_vec'][index])

#             current_retrieved_label_list.append(df_database['label'][index])

#         retrieved_visual_feature_embedding_cls_list.append(current_retrieved_visual_feature_embedding_cls_list)

#         retrieved_visual_feature_embedding_mean_list.append(current_retrieved_visual_feature_embedding_mean_list)

#         retrieved_textual_feature_embedding_list.append(current_retrieved_textual_feature_embedding_list)

#         retrieve_label_list.append(current_retrieved_label_list)

#     df_valid['retrieved_visual_feature_embedding_cls'] = retrieved_visual_feature_embedding_cls_list

#     df_valid['retrieved_visual_feature_embedding_mean'] = retrieved_visual_feature_embedding_mean_list

#     df_valid['retrieved_textual_feature_embedding'] = retrieved_textual_feature_embedding_list

#     df_valid['retrieved_label_list'] = retrieve_label_list

#     df_valid.to_pickle(valid_path)


def stack_retrieved_feature(train_path, valid_path, test_path, batch_size):
    # 加载数据
    df_train = pd.read_pickle(train_path)
    df_test = pd.read_pickle(test_path)
    df_valid = pd.read_pickle(valid_path)

    # 将所有数据框合并为一个
    # df_database = pd.concat([df_train, df_test, df_valid], axis=0).reset_index(drop=True)
    df_database = pd.concat([df_train, df_test, df_valid], axis=0).reset_index()

    # 创建从 image_id 到其索引的映射
    id_to_index = df_database.set_index('image_id')['index'].to_dict()

    # 定义根据 id_list 检索嵌入的函数
    def retrieve_embeddings(id_list):
        cls_vecs = []
        mean_vecs = []
        merged_text_vecs = []
        labels = []

        for item_id in id_list:
            index = id_to_index[item_id]  # 获取索引
            cls_vecs.append(df_database['cls_vec'][index])
            mean_vecs.append(df_database['mean_pooling_vec'][index])
            merged_text_vecs.append(df_database['merged_text_vec'][index])
            labels.append(df_database['label'][index])

        return cls_vecs, mean_vecs, merged_text_vecs, labels
    
    # 定义一个通用的处理函数，用于处理每个数据集
    def process_dataframe(df):
        retrieved_visual_feature_embedding_cls_list, retrieved_visual_feature_embedding_mean_list, \
        retrieved_textual_feature_embedding_list, retrieve_label_list = [], [], [], []

        for start in tqdm(range(0, len(df), batch_size)):
            end = min(start + batch_size, len(df))
            batch_id_list = df['retrieved_item_id'][start:end]

            for id_list in batch_id_list:
                cls_vecs, mean_vecs, merged_text_vecs, labels = retrieve_embeddings(id_list)
                retrieved_visual_feature_embedding_cls_list.append(cls_vecs)
                retrieved_visual_feature_embedding_mean_list.append(mean_vecs)
                retrieved_textual_feature_embedding_list.append(merged_text_vecs)
                retrieve_label_list.append(labels)

        df['retrieved_visual_feature_embedding_cls'] = retrieved_visual_feature_embedding_cls_list
        df['retrieved_visual_feature_embedding_mean'] = retrieved_visual_feature_embedding_mean_list
        df['retrieved_textual_feature_embedding'] = retrieved_textual_feature_embedding_list
        df['retrieved_label_list'] = retrieve_label_list

    # 处理训练数据
    process_dataframe(df_train)
    # ipdb.set_trace()
    df_train.to_pickle(train_path)
    
    # 处理测试数据
    process_dataframe(df_test)
    df_test.to_pickle(test_path)

    # 处理验证数据
    process_dataframe(df_valid)
    df_valid.to_pickle(valid_path)


def IB_pretrain_fusion(origin_path, z_path):
    df_ori = pd.read_pickle(origin_path)
    z = pd.read_pickle(z_path)
    # 初始化 compress_vec 列
    df_ori['compress_vec'] = None  
    for i in tqdm(range(len(df_ori)), desc="Processing rows"):
        df_ori.at[i, 'compress_vec'] = z[i].tolist()
    df_ori.to_pickle(origin_path)



# def label_log(data_path):
#     # 加载数据
#     df = pd.read_pickle(data_path)
#     # 对 'label' 列进行对数变换
#     df['label_log'] = [math.log2(df['label'][i] + 1) for i in range(len(df['label']))]
#     # 对 'retrieved_label' 列进行对数变换
#     df['retrieved_label'] = df['retrieved_label'].apply(lambda x: [math.log2(i + 1) for i in x])
#     # 对 'retrieved_label_list' 列进行对数变换
#     df['retrieved_label_list'] = df['retrieved_label_list'].apply(lambda x: [math.log2(i + 1) for i in x])
#     df.to_pickle(data_path)

def label_log(data_path):
    # 加载数据
    df = pd.read_pickle(data_path)
    # 对 'label' 列进行对数变换
    tqdm.pandas(desc="Processing label log")
    df['label'] = df['label'].progress_apply(lambda x: math.log2(x + 1))
    # 对 'retrieved_label' 列进行对数变换
    df['retrieved_label'] = df['retrieved_label'].progress_apply(lambda x: [math.log2(i + 1) for i in x])
    # 对 'retrieved_label_list' 列进行对数变换
    df['retrieved_label_list'] = df['retrieved_label_list'].progress_apply(lambda x: [math.log2(i + 1) for i in x])
    # 保存处理后的 DataFrame
    df.to_pickle(data_path)


if __name__ == "__main__":

    dataset_path = r'./datasets/Instagram/origin/dataset.pkl'
    data_path = r'./datasets/Instagram/origin/'
    train_path = r'./datasets/Instagram/origin/train.pkl'
    valid_path = r'./datasets/Instagram/origin/valid.pkl'
    test_path = r'./datasets/Instagram/origin/test.pkl'
    retrieval_pool_path = r'./datasets/Instagram/origin/retrieval_pool.pkl'
    retrieval_num = 700
    beta = 0.00005
    z_train_path = os.path.join(data_path, f'z_train_beta_{beta}.pkl')
    z_valid_path = os.path.join(data_path, f'z_valid_beta_{beta}.pkl')
    z_test_path = os.path.join(data_path, f'z_test_beta_{beta}.pkl')

    # create_retrieval_pool(train_path, valid_path, retrieval_pool_path)
    # print('Create retrieval pool done!')

    # retrieval_data(retrieval_num, train_path, retrieval_pool_path)
    # retrieval_data(retrieval_num, valid_path, retrieval_pool_path)
    # retrieval_data(retrieval_num, test_path, retrieval_pool_path)
    # print('Retrieval done!')

    # stack_retrieved_feature(train_path, valid_path, test_path, batch_size=10000)
    # print('Stack retrieved feature done!')

    # label_log(train_path)
    # label_log(valid_path)
    # label_log(test_path)
    # print('Label transform done!')

    IB_pretrain_fusion(train_path, z_train_path)
    IB_pretrain_fusion(valid_path, z_valid_path)
    IB_pretrain_fusion(test_path, z_test_path)
    print('IB fusion done!')

    # train_data = pd.read_pickle(train_path)
    # ipdb.set_trace()