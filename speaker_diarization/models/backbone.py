import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from torch import nn
from typing import Tuple

pretrained_settings = {}

models = [
    'resnet18', 'resnet34', 'resnet50',
    'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

input_sizes = {}
means = {}
stds = {}

for model_name in models:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

for model_name in ['inceptionv3']:
    input_sizes[model_name] = [3, 299, 299]
    means[model_name] = [0.5, 0.5, 0.5]
    stds[model_name] = [0.5, 0.5, 0.5]


for model_name in models:
    pretrained_settings[model_name] = {
        'imagenet': {
            'url': model_urls[model_name],
            'input_space': 'RGB',
            'input_size': input_sizes[model_name],
            'input_range': [0, 1],
            'mean': means[model_name],
            'std': stds[model_name],
            'num_classes': 1000
        }
    }


class ResNetEncoder(ResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        del self.fc

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')
        super().load_state_dict(state_dict, **kwargs)


class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                 middle_channels: Tuple[int, int, int],
                 kernel_size,
                 stride: Tuple[int, int] = (2, 2),
                 use_shortcut: bool = True):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels[0], kernel_size=(1, 1), stride=stride, bias=False),
            nn.BatchNorm2d(middle_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(middle_channels[0], middle_channels[1], kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels[1]),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(middle_channels[1], middle_channels[2], kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(middle_channels[2]),
        )
        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels[2], kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(middle_channels[2]),
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.block(x)
        out = self.block1(out)
        out = self.block2(out)

        out += self.shortcut(identity)

        out = self.activation(out)

        return out


class ResNet34s(nn.Module):
    def __init__(self, in_channels: int,
                 middle_channels: int):
        super(ResNet34s, self).__init__()
        self.in_channels = in_channels

        self.x1 = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=(7, 7), padding=2,  bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        self.x2 = nn.Sequential(
            ConvBlock(middle_channels, (48,  48, 96), kernel_size=3, stride=(1, 1)),
            ConvBlock(96, (48,  48, 96), kernel_size=3, use_shortcut=False, stride=(1, 1)),
        )

        self.x3 = nn.Sequential(
            ConvBlock(96, (96, 96, 128), kernel_size=3),
            ConvBlock(128, (96, 96, 128), kernel_size=3, stride=(1, 1),
                      use_shortcut=False),
            ConvBlock(128, (96, 96, 128), kernel_size=3, stride=(1, 1),
                      use_shortcut=False),
        )

        self.x4 = nn.Sequential(
            ConvBlock(128, (128, 128, 256), kernel_size=3),
            ConvBlock(256, (128, 128, 256), kernel_size=3, stride=(1, 1),
                      use_shortcut=False),
            ConvBlock(256, (128, 128, 256), kernel_size=3, stride=(1, 1),
                      use_shortcut=False),
        )

        # stride=(2, 1) or stride=(2, 2) ?
        self.x5 = nn.Sequential(
            ConvBlock(256, (256, 256, 512), kernel_size=3),
            ConvBlock(512, (256, 256, 512), kernel_size=3, stride=(1, 1),
                      use_shortcut=False),
            ConvBlock(512, (256, 256, 512), kernel_size=3, stride=(1, 1),
                      use_shortcut=False),
            nn.MaxPool2d((3, 1), stride=(2, 2))
        )

    def forward(self, x):

        x1 = self.x1(x)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)

        return [x5, x4, x3, x2, x1]


resnet_encoders = {
    'resnet18': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet18'],
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [2, 2, 2, 2],
        },
    },

    'resnet34s': {
        'encoder': ResNet34s,
        'out_shapes': (512, 256, 128, 96, 64),
        'params': {
            'in_channels': 1,
            'middle_channels': 64,
        },
    },

    'resnet34': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet34'],
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [3, 4, 6, 3],
        },
    },

    'resnet50': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet50'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
        },
    },

}

encoders = {}
encoders.update(resnet_encoders)


def get_encoder(name, encoder_weights=None):
    Encoder = encoders[name]['encoder']

    encoder = Encoder(**encoders[name]['params'])

    encoder.out_shapes = encoders[name]['out_shapes']

    if encoder_weights is not None:
        settings = encoders[name]['pretrained_settings'][encoder_weights]
        encoder.load_state_dict(model_zoo.load_url(settings['url']))

    return encoder
