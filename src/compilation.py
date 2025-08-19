from os import makedirs
from torch import load 
from torch.nn import Module
from torch.optim import Optimizer
from torchsystem import Depends
from torchsystem.compiler import Compiler, compile
from src.metrics import Metrics
from src.classifier import Classifier
from src.training import models, device, provider
from mltracker.ports import Models

compiler = Compiler[Classifier](provider=provider)
  
@compiler.step
def build_model(nn: Module, criterion: Module, optimizer: Module, metrics: Metrics, device: str = Depends(device)):
    print(f"Moving classifier to device {device}...")
    metrics.accuracy.to(device)
    metrics.loss.to(device)
    return Classifier(nn, criterion, optimizer, metrics).to(device)

@compiler.step
def bring_to_current_epoch(classifier: Classifier, models: Models = Depends(models)):
    print("Retrieving model from store...")
    model = models.read(classifier.id)
    if not model:
        print(f"model not found, creating one...")
        model = models.create(classifier.id, 'classifier')
    else:
        print(f"model found on epoch {model.epoch}")
    classifier.epoch = model.epoch
    return classifier 

@compiler.step
def restore_weights(classifier: Classifier): 
    if classifier.epoch != 0:    
        print("Restoring model weights") 
        makedirs(".data/weights", exist_ok=True)
        classifier.nn.load_state_dict(load(f"./data/weights/{classifier.name}-{classifier.hash}.pth", weights_only=True))
    return classifier

@compiler.step
def compile_model(classifier: Classifier):
    print("Compiling model...")
    return compile(classifier) 