import torch
from torch import nn
import torch.nn.functional as F
from .backbone import get_encoder


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


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    """Create and initialize a `nn.Conv1d` layer with spectral normalization."""
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    return torch.nn.utils.spectral_norm(conv)


class SelfAttention1d(nn.Module):
    """Self attention layer for nd."""
    def __init__(self, n_channels: int):
        super(SelfAttention1d, self).__init__()
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]), requires_grad=True)

    def forward(self, x):

        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


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

        #exp_cluster_score = torch.exp(cluster_score - max_cluster_score)

        #soft_assign = exp_cluster_score / (exp_cluster_score.sum(dim=1, keepdim=True))

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
                 loss_name: str = 'softmax',
                 use_attention: bool = False):
        super(Decoder, self).__init__()

        self.k_clusters = k_clusters
        self.mode = mode
        self.loss_name = loss_name
        self.g_clusters = g_clusters

        if self.mode != 'gvlad':
            self.g_clusters = 0

        self.use_attention = use_attention

        if use_attention:
            self.attention = SelfAttention1d(middle_channels)

        self.conv = nn.Sequential(nn.Conv2d(middle_channels, middle_channels, kernel_size=(7, 1), stride=(1, 1)),
                                  nn.ReLU(inplace=True))

        self.conv_center = nn.Conv2d(middle_channels, self.k_clusters + self.g_clusters, kernel_size=(7, 1), stride=(1, 1))

        self.vlad_polling = VladPooling(middle_channels, self.k_clusters, self.g_clusters, self.mode)

        self.fc = nn.Sequential(nn.Linear(middle_channels * self.k_clusters, middle_channels),
                                nn.BatchNorm1d(middle_channels),
                                nn.ReLU(inplace=True))

        self.logit = nn.Linear(middle_channels, out_channels, bias=False)

    def forward(self, features, **kwargs):

        if self.use_attention:

            features += self.attention(features)

        embeddings = self.vlad_polling((self.conv(features), self.conv_center(features)))

        embeddings = self.fc(embeddings)

        if 'asoftmax' in self.loss_name.lower():

            embeddings = F.normalize(embeddings, p=2, dim=-1)

        vlad = self.logit(embeddings)

        return embeddings, vlad


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


class SpeakerDiarization(EncoderDecoder):

    def __init__(
            self,
            middle_channels: int,
            out_channels: int,
            k_clusters: int = 8,
            g_clusters: int = 2,
            encoder_name: str = 'resnet34s',
            encoder_weights: str = None,
            mode: str = 'gvlad',
            loss_name: str = 'softmax',
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
                          loss_name,
                          use_attention)

        super().__init__(encoder, decoder, None)

        super().initialize()

        self.name = 'sp--{0}--{1}'.format(mode, encoder_name)