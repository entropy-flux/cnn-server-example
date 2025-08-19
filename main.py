from torch import Tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Dropout
from torchsystem.registry import register
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

class DMLP(Module):
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

class Digits:
    def __init__(self, train: bool, normalize: bool):
        self.transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]) if normalize else ToTensor()
        self.data = MNIST(download=True, root='./data/mnist', train=train)
    
    def __getitem__(self, index: int):   
        return self.transform(self.data[index][0]), self.data[index][1]
    
    def __len__(self):
        return len(self.data)


if __name__ == '__main__': 
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

    register(Adam, excluded_args=[0], excluded_kwargs={'params'})
    register(CrossEntropyLoss)
    register(Digits)
    register(DataLoader)
    register(DMLP)
    repository = getallmodels("MNIST-Digits", backend="tinydb")  

    training.provider.override(training.device, lambda: 'cuda') 
    training.provider.override(training.models, lambda: repository)
    training.producer.register(persistence.consumer)

    nn = DMLP(784, 512, 10, bias=True, p=0.4)
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
    } 

    for epoch in range(5):
        training.train(classifier, loaders['train'])
        training.evaluate(classifier, loaders['evaluation'])