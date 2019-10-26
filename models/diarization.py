import torch
from torch import nn
import torch.nn.functional as F
from .backbone import get_encoder
from .activation import Mish


def _get_activation(name: str):
    return nn.ReLU if name == 'relu' else Mish


class SpatialAttention2d(nn.Module):
    """Spatial Attention Layer"""
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class VladPooling(nn.Module):
    def __init__(self,
                 in_channels: int,
                 k_clusters: int = 8,
                 g_clusters: int = 2,
                 mode: str = 'gvlad'):

        super(VladPooling, self).__init__()
        self.k_clusters = k_clusters
        self.mode = mode
        self.g_clusters = g_clusters

        self.centroids = nn.Parameter(torch.randn(k_clusters + g_clusters, in_channels), requires_grad=True)
        self._init_params()

    def _init_params(self):
        nn.init.orthogonal_(self.centroids.data)

    def forward(self, x):

        features, cluster_score = x

        num_features = features.shape[1]

        max_cluster_score = cluster_score.max(dim=0, keepdim=True)[0]
        exp_cluster_score = torch.exp(cluster_score - max_cluster_score)

        soft_assign = exp_cluster_score / exp_cluster_score.sum(dim=0, keepdim=True)

        residual_features = features.unsqueeze(1)

        residual_features = residual_features - self.centroids.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        soft_assign = soft_assign.unsqueeze(2)

        weighted_res = soft_assign * residual_features

        cluster_res = torch.sum(weighted_res, dim=(3, 4), keepdim=False)

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, :self.k_clusters, :]

        cluster_l2 = F.normalize(cluster_res, p=2, dim=-1)
        outputs = cluster_l2.reshape(-1, (int(self.k_clusters) * int(num_features)))

        return outputs


class SpeakerDiarization(nn.Module):

    def __init__(self,
                 middle_channels: int,
                 out_channels: int,
                 k_clusters: int = 10,
                 g_clusters: int = 2,
                 encoder_name: str = 'resnet34s',
                 weights: str = None,
                 mode: str = 'gvlad',
                 activation_name: str = 'relu',
                 loss_name: str = 'softmax',
                 use_attention: bool = False):
        super(SpeakerDiarization, self).__init__()

        self.k_clusters = k_clusters
        self.mode = mode
        self.loss_name = loss_name
        self.g_clusters = g_clusters

        if self.mode != 'gvlad':
            self.g_clusters = 0

        self.encoder = get_encoder(encoder_name, weights, activation_name=activation_name)

        self.use_attention = use_attention

        if use_attention:
            self.attention = SpatialAttention2d(middle_channels)

        self.conv = nn.Sequential(nn.Conv2d(middle_channels, middle_channels, kernel_size=(7, 1), stride=(1, 1)),
                                _get_activation(activation_name)(inplace=True))

        self.conv_center = nn.Conv2d(middle_channels, self.k_clusters + self.g_clusters, kernel_size=(7, 1), stride=(1, 1))

        self.vlad_polling = VladPooling(middle_channels, self.k_clusters, self.g_clusters, self.mode)

        self.fc = nn.Sequential(nn.Linear(middle_channels * self.k_clusters, middle_channels),
                                nn.BatchNorm1d(middle_channels),
                                _get_activation(activation_name)(inplace=True))

        self.logit = nn.Linear(middle_channels, out_channels, bias=False)

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize kernels of Conv2d layers as kaiming normal
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features, **kwargs):

        x = self.encoder(features)

        if self.use_attention:

            x += self.attention(x)

        embeddings = self.vlad_polling((self.conv(x), self.conv_center(x)))

        vlad = self.fc(embeddings)

        if 'asoftmax' in self.loss_name.lower():

            vlad = F.normalize(vlad, p=2, dim=-1)

        vlad = self.logit(vlad)

        return embeddings, vlad

