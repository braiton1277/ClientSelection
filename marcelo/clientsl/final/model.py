import torch.nn as nn
from torchvision.models import resnet18


class SmallCNN(nn.Module):
    """ResNet18 adaptada para CIFAR-10 (32x32) com BatchNorm padrão."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        self.net = m

    def forward(self, x):
        return self.net(x)
