import torch.nn as nn
from typing import Any

# Embedding net for data visualization
# Network architecture should be same as Siamese network
class EmbeddingNet(nn.Module):

    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.features = _make_layers(cfg['VGG10'])
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def __call__(self, *input, **kwargs) -> Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):
        x = x.float()
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def get_embeddings(self, x):
        return self.forward(x)

# Siamense network for train
class EmbeddingNetSiamese(nn.Module):

    def __init__(self):
        super(EmbeddingNetSiamese, self).__init__()
        self.features = _make_layers(cfg['VGG10'])
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def __call__(self, *input, **kwargs) -> Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def siamese_get_embeddings(self, input1, input2):
        output1 = self.forward(input1)
        output2 = self.forward(input2)
        return output1, output2

# build neural network
def _make_layers(cfg):
    layers = []
    in_channels = 3
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


cfg = {
    'VGG7': [64, 'M', 128, 'M', 128, 'M', 256, 'M'],
    'VGG8': [64, 'M', 128, 'M', 256, 'M', 256, 256, 'M'],
    'VGG9': [64, 'M', 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGG10': [64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M'],
    'VGG15': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
}


