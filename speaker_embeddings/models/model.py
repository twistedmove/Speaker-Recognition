import torch
from torch import nn
import torch.nn.functional as F
from .backbone import get_encoder


# Source: https://github.com/luuuyi/CBAM.PyTorch
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# Source: https://github.com/luuuyi/CBAM.PyTorch
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvolutionalBlockAttentionModule(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ConvolutionalBlockAttentionModule, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out


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

        max_cluster_score = cluster_score.max(dim=1, keepdim=True)[0]

        soft_assign = F.softmax(cluster_score - max_cluster_score, dim=1)

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


class Decoder(nn.Module):

    def __init__(self,
                 middle_channels: int,
                 out_channels: int,
                 k_clusters: int = 8,
                 g_clusters: int = 2,
                 mode: str = 'gvlad',
                 use_attention: bool = False):
        super(Decoder, self).__init__()

        self.k_clusters = k_clusters
        self.mode = mode
        self.g_clusters = g_clusters

        if self.mode != 'gvlad':
            self.g_clusters = 0

        self.attention = nn.Identity()

        if use_attention:
            self.attention = ConvolutionalBlockAttentionModule(middle_channels, kernel_size=7)

        self.conv = nn.Sequential(nn.Conv2d(middle_channels, middle_channels, kernel_size=(7, 1), stride=(1, 1)),
                                  nn.ReLU(inplace=True))

        self.conv_center = nn.Conv2d(middle_channels, self.k_clusters + self.g_clusters, kernel_size=(7, 1), stride=(1, 1))

        self.vlad_polling = VladPooling(middle_channels, self.k_clusters, self.g_clusters, self.mode)

        self.fc = nn.Sequential(nn.Linear(middle_channels * self.k_clusters, middle_channels, bias=True),
                                nn.ReLU(inplace=True))

        self.logit = nn.Linear(middle_channels, out_channels, bias=False)

    def forward(self, features, **kwargs):

        x5, x4, x3, x2, x1 = features

        features = self.attention(x5)

        conv = self.conv(features)

        conv_center = self.conv_center(features)

        embeddings = self.vlad_polling((conv, conv_center))

        embeddings = self.fc(embeddings)

        logit = self.logit(embeddings)

        return embeddings, logit


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EncoderDecoder(Model):

    def __init__(self, encoder, decoder, activation):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, features):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        x = self.encoder(features)
        x = self.decoder(x)
        return x

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)

        return x


class SpeakerRecognition(EncoderDecoder):

    def __init__(
            self,
            middle_channels: int,
            out_channels: int,
            k_clusters: int = 8,
            g_clusters: int = 2,
            encoder_name: str = 'resnet34s',
            encoder_weights: str = None,
            mode: str = 'gvlad',
            use_attention: bool = False,
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = Decoder(middle_channels,
                          out_channels,
                          k_clusters,
                          g_clusters,
                          mode,
                          use_attention)

        super().__init__(encoder, decoder, None)

        super().initialize()

        self.name = 'sp--{0}--{1}'.format(mode, encoder_name)
