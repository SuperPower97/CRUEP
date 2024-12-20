import torch.utils.data
import pandas as pd
from torch_geometric.data import Dataset, HeteroData


def custom_collate_fn_ib(batch):
    mean_pooling_vec, merge_text_vec, compress_vec, retrieved_visual_feature_embedding_cls, \
        retrieved_textual_feature_embedding, retrieved_label_list, label = zip(*batch)
    return torch.tensor(mean_pooling_vec, dtype=torch.float32), \
        torch.tensor(merge_text_vec, dtype=torch.float32), \
        torch.tensor(compress_vec, dtype=torch.float32), \
        torch.tensor(retrieved_visual_feature_embedding_cls, dtype=torch.float32), \
        torch.tensor(retrieved_textual_feature_embedding, dtype=torch.float32), \
        torch.tensor(retrieved_label_list, dtype=torch.float32), \
        torch.tensor(label, dtype=torch.float32).unsqueeze(-1)

def custom_collate_fn(batch):
    mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
        retrieved_textual_feature_embedding, retrieved_label_list, label = zip(*batch)
    return torch.tensor(mean_pooling_vec, dtype=torch.float32), \
        torch.tensor(merge_text_vec, dtype=torch.float32), \
        torch.tensor(retrieved_visual_feature_embedding_cls, dtype=torch.float32), \
        torch.tensor(retrieved_textual_feature_embedding, dtype=torch.float32), \
        torch.tensor(retrieved_label_list, dtype=torch.float32), \
        torch.tensor(label, dtype=torch.float32).unsqueeze(-1)

def pre_custom_collate_fn(batch):
    mean_pooling_vec, merge_text_vec, label = zip(*batch)
    return torch.tensor(mean_pooling_vec, dtype=torch.float32), \
        torch.tensor(merge_text_vec, dtype=torch.float32), \
        torch.tensor(label, dtype=torch.float32).unsqueeze(-1)


class MyData(torch.utils.data.Dataset):

    def __init__(self, retrieval_num, path):
        super().__init__()

        self.path = path
        self.retrieval_num = retrieval_num
        self.dataframe = pd.read_pickle(path)
        self.label = self.dataframe['label']
        self.mean_pooling_vec = self.dataframe['mean_pooling_vec']
        self.merge_text_vec = self.dataframe['merged_text_vec']
        self.retrieval_visual_feature_embedding_cls = self.dataframe['retrieved_visual_feature_embedding_cls']
        self.retrieval_textual_feature_embedding = self.dataframe['retrieved_textual_feature_embedding']
        self.retrieval_label_list = self.dataframe['retrieved_label_list']
        self.RRCP = self.dataframe['RRCP_silver']

    def __getitem__(self, item):

        label = self.label[item]
        mean_pooling_vec = self.mean_pooling_vec[item]
        merge_text_vec = self.merge_text_vec[item]
        retrieved_visual_feature_embedding_cls = self.retrieval_visual_feature_embedding_cls[item]
        retrieved_textual_feature_embedding = self.retrieval_textual_feature_embedding[item]
        retrieved_label_list = self.retrieval_label_list[item]
        RRCP = self.RRCP[item]

        return mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
            retrieved_textual_feature_embedding, retrieved_label_list, RRCP, label

    def __len__(self):
        return len(self.dataframe)


class Origin_data(torch.utils.data.Dataset):

    def __init__(self, path):
        super().__init__()

        self.path = path
        self.dataframe = pd.read_pickle(path)
        self.label = self.dataframe['label']
        # image info
        self.mean_pooling_vec = self.dataframe['mean_pooling_vec']  
        # merged_text info
        self.merge_text_vec = self.dataframe['merged_text_vec']
        # meta info
        
    def __getitem__(self, item):

        label = self.label[item]
        mean_pooling_vec = self.mean_pooling_vec[item]
        merge_text_vec = self.merge_text_vec[item]

        return mean_pooling_vec, merge_text_vec, label

    def __len__(self):
        return len(self.dataframe)


class OwnData_IB(torch.utils.data.Dataset):

    def __init__(self, retrieval_num, path):
        super().__init__()

        self.path = path
        self.retrieval_num = retrieval_num
        self.dataframe = pd.read_pickle(path)
        self.label = self.dataframe['label']
        self.mean_pooling_vec = self.dataframe['mean_pooling_vec']
        self.merge_text_vec = self.dataframe['merged_text_vec']
        self.compress_vec = self.dataframe['compress_vec']
        self.retrieval_visual_feature_embedding_cls = self.dataframe['retrieved_visual_feature_embedding_cls']
        self.retrieval_textual_feature_embedding = self.dataframe['retrieved_textual_feature_embedding']
        self.retrieval_label_list = self.dataframe['retrieved_label_list']
        

    def __getitem__(self, item):

        label = self.label[item]
        mean_pooling_vec = self.mean_pooling_vec[item]
        merge_text_vec = self.merge_text_vec[item]
        compress_vec = self.compress_vec[item]
        retrieved_visual_feature_embedding_cls = self.retrieval_visual_feature_embedding_cls[item]
        retrieved_textual_feature_embedding = self.retrieval_textual_feature_embedding[item]
        retrieved_label_list = self.retrieval_label_list[item]
        return mean_pooling_vec, merge_text_vec, compress_vec, retrieved_visual_feature_embedding_cls, \
            retrieved_textual_feature_embedding, retrieved_label_list, label

    def __len__(self):
        return len(self.dataframe)


class OwnData(torch.utils.data.Dataset):

    def __init__(self, retrieval_num, path):
        super().__init__()

        self.path = path
        self.retrieval_num = retrieval_num
        self.dataframe = pd.read_pickle(path)
        self.label = self.dataframe['label']
        self.mean_pooling_vec = self.dataframe['mean_pooling_vec']
        self.merge_text_vec = self.dataframe['merged_text_vec']
        self.retrieval_visual_feature_embedding_cls = self.dataframe['retrieved_visual_feature_embedding_cls']
        self.retrieval_textual_feature_embedding = self.dataframe['retrieved_textual_feature_embedding']
        self.retrieval_label_list = self.dataframe['retrieved_label_list']
        

    def __getitem__(self, item):

        label = self.label[item]
        mean_pooling_vec = self.mean_pooling_vec[item]
        merge_text_vec = self.merge_text_vec[item]
        retrieved_visual_feature_embedding_cls = self.retrieval_visual_feature_embedding_cls[item]
        retrieved_textual_feature_embedding = self.retrieval_textual_feature_embedding[item]
        retrieved_label_list = self.retrieval_label_list[item]
        return mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, retrieved_textual_feature_embedding, retrieved_label_list, label

    def __len__(self):
        return len(self.dataframe)





    


