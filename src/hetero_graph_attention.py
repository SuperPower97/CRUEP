import torch
import torch.nn as nn
import ipdb
from heterograph_variable_length import GraphLearner, GraphLearner_IB


class Model_IB(nn.Module):

    def __init__(self, retrieval_num, feature_dim, hidden_dim):

        super(Model_IB, self).__init__()
        self.predict_linear_1 = nn.Linear(hidden_dim, hidden_dim)  # (768, 768)
        self.predict_linear_2 = nn.Linear(hidden_dim * 2, 1)        # (1536, 768)
        self.label_embedding_linear = nn.Linear(retrieval_num, hidden_dim)
        self.graph = GraphLearner_IB(device='cuda:0', feature_dim=feature_dim, hidden_dim=hidden_dim)
        

    def forward(self, retrieved_label_list,
                mean_pooling_vec, merge_text_vec, compress_vec, 
                retrieved_visual_feature_embedding_cls,
                retrieved_textual_feature_embedding):
        hetero_emb = self.graph(merge_text_vec, mean_pooling_vec, compress_vec, retrieved_textual_feature_embedding, retrieved_visual_feature_embedding_cls)
        
        output = self.predict_linear_1(hetero_emb)

        label = self.label_embedding_linear(retrieved_label_list)
        
        output = torch.cat([output, label], dim=1)

        output = self.predict_linear_2(output)

        return output


class Model(nn.Module):

    def __init__(self, retrieval_num, feature_dim, hidden_dim):

        super(Model, self).__init__()
        self.predict_linear_1 = nn.Linear(hidden_dim, hidden_dim)  # (768, 768)
        self.predict_linear_2 = nn.Linear(hidden_dim * 2, 1)        # (1536, 768)
        self.label_embedding_linear = nn.Linear(retrieval_num, hidden_dim)
        self.graph = GraphLearner(device='cuda:0', feature_dim=feature_dim, hidden_dim=hidden_dim)
        

    def forward(self, retrieved_label_list,
                mean_pooling_vec, merge_text_vec,
                retrieved_visual_feature_embedding_cls,
                retrieved_textual_feature_embedding):
        hetero_emb = self.graph(merge_text_vec, mean_pooling_vec, retrieved_textual_feature_embedding, retrieved_visual_feature_embedding_cls)
        
        output = self.predict_linear_1(hetero_emb)

        label = self.label_embedding_linear(retrieved_label_list)
        
        output = torch.cat([output, label], dim=1)

        output = self.predict_linear_2(output)

        return output
