from torch import Tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Dropout
from torchsystem.registry import register
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

class MLP(Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, bias: bool, p: float):
        super().__init__()
        self.input_layer = Linear(input_size, hidden_size, bias=bias)
        self.dropout = Dropout(p)
        self.activation = ReLU()
        self.output_layer = Linear(hidden_size, output_size, bias=bias)

    def forward(self, features: Tensor) -> Tensor:
        features = self.input_layer(features)
        features = self.dropout(features)
        features = self.activation(features)
        return self.output_layer(features)  