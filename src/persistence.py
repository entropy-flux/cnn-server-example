from mltracker.ports import Models
from torch import save
from torchsystem import Depends
from torchsystem.services import Consumer
from src.training import provider, models
from src.training import Trained, Evaluated 

consumer = Consumer(provider=provider)

@consumer.handler
def bump_epoch(event: Trained):
    event.model.epoch += 1 

@consumer.handler
def save_epoch(event: Trained, models: Models = Depends(models)):
    model = models.read(event.model.hash) or models.create(event.model.hash) 
    model.epoch += 1

@consumer.handler
def print_metrics(event: Trained | Evaluated):
    print(f"-----------------------------------------------------------------")
    print(f"Epoch: {event.model.epoch}, Average loss: {event.results['loss'].item()}, Average accuracy: {event.results['accuracy'].item()}")
    print(f"-----------------------------------------------------------------")

@consumer.handler
def handle_results(event: Trained | Evaluated, models: Models = Depends(models)):
    model = models.read(event.model.id)
    for name, metric in event.results.items():
        model.metrics.add(name, metric.item(), event.model.epoch, event.model.phase) 
 
def persist_model(event: Trained):
    save(event.model.nn.state_dict(), f"./data/weights/{event.model.name}-{event.model.hash}.pth")