from torch import Tensor
from torch.nn import Module, Conv2d, Linear, ReLU, Dropout, Flatten
from torchsystem.registry import register
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

class CNN(Module):
    def __init__(self, in_channels: int, hidden_channels: int, output_size: int, bias: bool, p: float):
        super().__init__() 
        self.convolutional_layer = Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.dropout = Dropout(p)
        self.activation = ReLU()
        self.flatten = Flatten() 
        self.output_layer = Linear(hidden_channels * 28 * 28, output_size, bias=bias)

    def forward(self, features: Tensor) -> Tensor:
        features = self.convolutional_layer(features)          
        features = self.dropout(features)
        features = self.activation(features)
        features = self.flatten(features)         
        return self.output_layer(features) 
