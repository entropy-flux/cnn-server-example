from torch import Tensor
from torch import reshape
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets.mnist import MNIST
from torcheval.metrics import Mean, MulticlassAccuracy 

class Metrics:
    def __init__(self, device: str | None = None): 
        self.accuracy = MulticlassAccuracy(num_classes=10, device=device)
        
    def update(self, predictions: Tensor, targets: Tensor) -> None: 
        self.accuracy.update(predictions, targets)
        
    def compute(self) -> dict[str, Tensor]:
        return { 
            'accuracy': self.accuracy.compute()
        }
    
    def reset(self) -> None: 
        self.accuracy.reset() 


if __name__ == "__main__":  
    from pytannic.client import Client
    from pytannic.torch.tensors import serialize, deserialize

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    mnist = MNIST(root="./data", train=False, download=True, transform=transform)
    loader = DataLoader(dataset=mnist, batch_size=256)
    metrics = Metrics(device="cpu")
     
    with Client("127.0.0.1", 8080) as client:
        for batch_idx, (images, labels) in enumerate(loader): 
            data = serialize(images.view(images.size(0), -1))
            client.send(data) 
            predictions = deserialize(client.receive())
            metrics.update(predictions, labels)
            results = metrics.compute()
            print(f"Accuracy: {results['accuracy'].item():.4f}") 
        print("Done!")