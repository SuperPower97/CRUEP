import torch
import torch.nn as nn
from hetero_graph_attention import Model_IB as hetero_graph_attention_ib
from hetero_graph_attention import Model as hetero_graph_attention
import time


class model_prediction_IB(nn.Module):

    def __init__(self, retrieval_num, feature_dim, hidden_dim=768):
        super(model_prediction_IB, self).__init__()
        self.retrieval_num = retrieval_num
        self.hetero_graph_attention = hetero_graph_attention_ib(retrieval_num, feature_dim, hidden_dim)


    def forward(self, mean_pooling_vec, merge_text_vec, compress_vec, retrieved_visual_feature_embedding_cls,
                retrieved_textual_feature_embedding, retrieved_label_list):
        retrieved_visual_feature_embedding_cls = retrieved_visual_feature_embedding_cls.squeeze(2)
        retrieved_textual_feature_embedding = retrieved_textual_feature_embedding.squeeze(2)

        retrieved_textual_feature_embedding = retrieved_textual_feature_embedding[:, :self.retrieval_num, :]
        retrieved_visual_feature_embedding_cls = retrieved_visual_feature_embedding_cls[:, :self.retrieval_num, :]
        retrieved_label_list = retrieved_label_list[:, :self.retrieval_num]

        output = self.hetero_graph_attention(retrieved_label_list, 
                                      mean_pooling_vec, merge_text_vec, compress_vec,
                                      retrieved_visual_feature_embedding_cls,
                                      retrieved_textual_feature_embedding)

        return output


class model_prediction(nn.Module):

    def __init__(self, retrieval_num, feature_dim, hidden_dim=768):
        super(model_prediction, self).__init__()
        self.retrieval_num = retrieval_num
        self.hetero_graph_attention = hetero_graph_attention(retrieval_num, feature_dim, hidden_dim)


    def forward(self, mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls,
                retrieved_textual_feature_embedding, retrieved_label_list):
        retrieved_visual_feature_embedding_cls = retrieved_visual_feature_embedding_cls.squeeze(2)
        retrieved_textual_feature_embedding = retrieved_textual_feature_embedding.squeeze(2)

        retrieved_textual_feature_embedding = retrieved_textual_feature_embedding[:, :self.retrieval_num, :]
        retrieved_visual_feature_embedding_cls = retrieved_visual_feature_embedding_cls[:, :self.retrieval_num, :]
        retrieved_label_list = retrieved_label_list[:, :self.retrieval_num]

        output = self.hetero_graph_attention(retrieved_label_list, 
                                      mean_pooling_vec, merge_text_vec, 
                                      retrieved_visual_feature_embedding_cls,
                                      retrieved_textual_feature_embedding)

        return output  
