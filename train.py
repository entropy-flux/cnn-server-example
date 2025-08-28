from torch import Tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Dropout
from torch.nn import Conv2d, Flatten
from torchsystem.registry import register
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize  
 
class Digits:
    def __init__(self, train: bool, normalize: bool):
        self.transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]) if normalize else ToTensor()
        self.data = MNIST(download=True, root='./data/mnist', train=train)
    
    def __getitem__(self, index: int):   
        return self.transform(self.data[index][0]), self.data[index][1]
    
    def __len__(self):
        return len(self.data) 
    

class CNN(Module):
    def __init__(self, in_channels: int, hidden_channels: int, output_size: int, bias: bool, p: float):
        super().__init__() 
        self.convolutional_layer = Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.dropout = Dropout(p)
        self.activation = ReLU()
        self.flatten = Flatten() 
        self.output_layer = Linear(hidden_channels * 28 * 28, output_size, bias=bias)

    def forward(self, features: Tensor) -> Tensor:
        features = features.view(features.size(0), 1, 28, 28)
        features = self.convolutional_layer(features)          
        features = self.dropout(features)
        features = self.activation(features)
        features = self.flatten(features)         
        return self.output_layer(features)  


if __name__ == '__main__': 
    from torch import cuda
    from torch.nn import CrossEntropyLoss
    from torch.nn import Dropout
    from torch.optim import Adam
    from torch.utils.data import DataLoader 
    from torchsystem.registry import gethash
    from mltracker import getallmodels
    from src import training 
    from src import persistence 
    from src.metrics import Metrics
    from src.compilation import compiler 

    from pytannic.torch.modules import write

    register(Adam, excluded_args=[0], excluded_kwargs={'params'})
    register(CrossEntropyLoss)
    register(Digits) 
    register(CNN) 

    training.provider.override(training.device, lambda: 'cuda' if cuda.is_available() else 'cpu') 
    training.provider.override(training.models, lambda: getallmodels("MNIST-Digits", backend="tinydb")  )
    training.producer.register(persistence.consumer)
 
    nn = CNN(1, 32, 10, bias=True, p=0.2)  
    criterion = CrossEntropyLoss()
    optimizer = Adam(nn.parameters(), lr=0.001)
    metrics = Metrics()
    classifier = compiler.compile(nn, criterion, optimizer, metrics)
    datasets = {
        'train': Digits(train=True, normalize=True),
        'evaluation': Digits(train=False,  normalize=True),
    }


    loaders = {
        'train': DataLoader(datasets['train'], batch_size=256, shuffle=True, pin_memory=True, pin_memory_device='cuda:0', num_workers=4),
        'evaluation': DataLoader(datasets['evaluation'], batch_size=256, shuffle=False, pin_memory=True, pin_memory_device='cuda:0', num_workers=4) 
    } if cuda.is_available() else {
        'train': DataLoader(datasets['train'], batch_size=256, shuffle=True),
        'evaluation': DataLoader(datasets['evaluation'], batch_size=256, shuffle=False) 
    }

    for epoch in range(15):
        training.train(classifier, loaders['train'])
        training.evaluate(classifier, loaders['evaluation'])

    write(classifier.nn, f"data/tannic/{classifier.name}")