import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import ipdb


class Encoder(nn.Module):
    def __init__(self, feature_dim ,z_dim):
        super(Encoder, self).__init__()

        self.z_dim = z_dim

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, z_dim * 2),  # Output mean and stddev
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)

        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = F.softplus(sigma) + 1e-7  # Ensure sigma is positive

        return Independent(Normal(loc=mu, scale=sigma), 1)  # Return a factorized Normal distribution


class Decoder(nn.Module):
    def __init__(self, z_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(z_dim, output_dim)

    def forward(self, z):
        return self.fc(z)


class IB_Regressor(nn.Module):

    def __init__(self, feature_dim, hidden_dim, alpha=0.5, frame_num=1):

        super(IB_Regressor, self).__init__()
        self.encoder = Encoder(feature_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, 1)
        self.prior = Normal(torch.zeros(hidden_dim).to('cuda:0'), torch.ones(hidden_dim).to('cuda:0')) # Standard normal prior

        self.alpha = alpha
        self.feature_dim = feature_dim
        self.predict_linear_1 = nn.Linear(feature_dim * 2, hidden_dim)
        self.predict_linear_2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8, batch_first=True)


    def forward(self, mean_pooling_vec, merge_text_vec):
        mean_pooling_vec = mean_pooling_vec.squeeze(1)
        merge_text_vec = merge_text_vec.squeeze(1)
        packed_x = torch.cat([mean_pooling_vec, merge_text_vec], dim=1)
        x, _ = self.multihead_attn(packed_x, packed_x, packed_x)

        z_dist = self.encoder(x)
        z = z_dist.rsample()  # Reparameterization trick
        output = self.decoder(z)
        
        return z, output, z_dist
    
    def kl_divergence(self, z_dist):
        # 计算 KL 散度 KL(P(Z|X) || P(Z))
        mu_q = z_dist.base_dist.loc  # 获取均值
        sigma_q = z_dist.base_dist.scale  # 获取标准差
        mu_p = self.prior.loc
        sigma_p = self.prior.scale
        # 计算 KL 散度
        kl = torch.sum(sigma_q.log() - sigma_p.log() + (sigma_p ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_q ** 2) - 0.5)
        return kl