import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import ipdb
from torch.autograd import Variable
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv
from torch_geometric.data import HeteroData

### rewrite ###
class HeteroGNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HeteroGNN, self).__init__()
        # 定义线性层以对齐特征维度
        self.user_linear = nn.Linear(in_channels, out_channels)
        self.conv1 = HeteroConv({
            ('image', 'to', 'user'): SAGEConv(out_channels, out_channels),  # 图像到用户的卷积
            ('text', 'to', 'user'): SAGEConv(out_channels, out_channels)   # 文本到用户的卷积
        })
        self.relu = nn.ReLU()
        # self.conv2 = HeteroConv({
        #     ('image', 'to', 'user'): GCNConv(in_channels, out_channels, add_self_loops=False),  # 图像到用户的卷积
        #     ('text', 'to', 'user'): SAGEConv(in_channels, out_channels)    # 文本到用户的卷积
        # })

    def forward(self, x_dict, edge_index_dict):
        # 线性转换维度对齐
        x_dict['user'] = self.user_linear(x_dict['user'])   # (batch_size, hidden_dim * 2) ——> (batch_size, hidden_dim)
        # 第一个卷积层
        output_dict = self.conv1(x_dict, edge_index_dict) 
        user_output = output_dict['user']    
        # 非线性层
        user_output = self.relu(user_output)
        return user_output


class GraphLearner_IB(nn.Module):
    def __init__(self, device, feature_dim, hidden_dim):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.heteroGNN = HeteroGNN(self.feature_dim, self.hidden_dim)

    def forward(self, input_text, input_img, input_compress, base_text_features, base_img_features):
        batch_size, max_nodes, hidden_dim = base_text_features.shape
        
        with torch.no_grad():
            node_cluster_t = base_text_features.reshape(-1, hidden_dim)   # (batch_size * max_nodes, hidden_dim)
            node_cluster_i = base_img_features.reshape(-1, hidden_dim)    # (batch_size * max_nodes, hidden_dim)
            input_img = input_img.squeeze(1)
            input_text = input_text.squeeze(1)
        # ipdb.set_trace()
        for index in range(1):
            with torch.no_grad():
                data = HeteroData()

                input_feat = torch.cat([input_text, input_img, input_compress], dim=1)  # (batch, feature_dim)
                data['user'].x = input_feat
                data['image'].x = node_cluster_i                        # (batch_size * max_nodes, hidden_dim)
                data['text'].x = node_cluster_t                         # (batch_size * max_nodes, hidden_dim)
                # 创建用户到图像的边
                user_to_image_edges = torch.tensor(
                    [
                    [j + (i * max_nodes) for i in range(batch_size) for j in range(max_nodes)],     # 图像节点索引
                    [i for i in range(batch_size) for _ in range(max_nodes)],  # 用户索引
                    ], 
                    dtype=torch.long
                )
        
                # 创建用户到文本的边
                user_to_text_edges = torch.tensor(
                    [
                    [j + (i * max_nodes) for i in range(batch_size) for j in range(max_nodes)],     # 文本节点索引
                    [i for i in range(batch_size) for _ in range(max_nodes)],  # 用户索引
                    ],  
                    dtype=torch.long
                )

                # 添加边到数据
                data['image', 'to', 'user'].edge_index = user_to_image_edges
                data['text', 'to', 'user'].edge_index = user_to_text_edges
                data.to('cuda:0')
        # ipdb.set_trace()        
        graph_out = self.heteroGNN(data.x_dict, data.edge_index_dict)

        return graph_out


class GraphLearner(nn.Module):
    def __init__(self, device, feature_dim, hidden_dim):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.heteroGNN = HeteroGNN(self.feature_dim, self.hidden_dim)

    def forward(self, input_text, input_img, base_text_features, base_img_features):
        batch_size, max_nodes, hidden_dim = base_text_features.shape
        
        with torch.no_grad():
            node_cluster_t = base_text_features.reshape(-1, hidden_dim)   # (batch_size * max_nodes, hidden_dim)
            node_cluster_i = base_img_features.reshape(-1, hidden_dim)    # (batch_size * max_nodes, hidden_dim)
            input_img = input_img.squeeze(1)
            input_text = input_text.squeeze(1)
        # ipdb.set_trace()
        for index in range(1):
            with torch.no_grad():
                data = HeteroData()

                input_feat = torch.cat([input_text, input_img], dim=1)  # (batch, feature_dim)
                data['user'].x = input_feat
                data['image'].x = node_cluster_i                        # (batch_size * max_nodes, hidden_dim)
                data['text'].x = node_cluster_t                         # (batch_size * max_nodes, hidden_dim)
                # 创建用户到图像的边
                user_to_image_edges = torch.tensor(
                    [
                    [j + (i * max_nodes) for i in range(batch_size) for j in range(max_nodes)],     # 图像节点索引
                    [i for i in range(batch_size) for _ in range(max_nodes)],  # 用户索引
                    ], 
                    dtype=torch.long
                )
        
                # 创建用户到文本的边
                user_to_text_edges = torch.tensor(
                    [
                    [j + (i * max_nodes) for i in range(batch_size) for j in range(max_nodes)],     # 文本节点索引
                    [i for i in range(batch_size) for _ in range(max_nodes)],  # 用户索引
                    ],  
                    dtype=torch.long
                )

                # 添加边到数据
                data['image', 'to', 'user'].edge_index = user_to_image_edges
                data['text', 'to', 'user'].edge_index = user_to_text_edges
                data.to('cuda:0')
        # ipdb.set_trace()        
        graph_out = self.heteroGNN(data.x_dict, data.edge_index_dict)

        return graph_out

