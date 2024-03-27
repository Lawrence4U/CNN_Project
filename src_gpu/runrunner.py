from cnn import CNN
import torchvision
from cnn import load_data
from cnn import load_model_weights
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import wandb
import random
import os

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer

def build_criterion(criterion):
    return nn.CrossEntropyLoss()

def run_model(config=None):
    with wandb.init(config=config):
        config = wandb.config
        print(config)
        train_dir = 'dataset/training'
        valid_dir = 'dataset/validation'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_loader, valid_loader, num_classes = load_data(train_dir, 
                                                            valid_dir, 
                                                            batch_size=config.batch_size, 
                                                            img_size=224) # ResNet50 requires 224x224 images
        model = CNN(torchvision.models.resnet50(weights=config.weights), num_classes, unfreezed_layers=config.unfrozen_layers).to(device)

        optimizer = build_optimizer(model,config.optimizer,config.learning_rate)
        criterion = build_criterion(config.criterion)
        history = model.train_model(train_loader, valid_loader, optimizer, criterion, epochs=config.epochs)
        train_loss = history["train_loss"][-1]
        train_acc = history["train_accuracy"][-1]
        valid_loss = history["valid_loss"][-1]
        valid_acc = history["valid_accuracy"][-1]


        wandb.log({"train_acc": train_acc,
                "train_loss": train_loss,
                "valid_acc": valid_acc,
                "valid_loss": valid_loss})

sweep_configuration = {
    "name": "sweepdemo",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "valid_loss"},
    "parameters": {
        "learning_rate": {"value": 1e-4},
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10]},
        "optimizer": {"value": "adam"},
        "model": {"value": "resnet50"},
        "weights": {"value": "DEFAULT"},
        "criterion": {"value": "CrossEntropyLoss"},
        "unfrozen_layers": {"value": 0},
    },
}

sweep_id = wandb.sweep(sweep_configuration, project="pytorch-sweeps-demo")
wandb.agent(sweep_id, run_model)
wandb.finish()

# config={
#     "model": "resnet50",
#     "epochs": 2,
#     "weights": "DEFAULT",
#     "batch_size": 32,
#     "loss_func": "CrossEntropyLoss",
#     "unfrozen_layers": 0
# }


# wandb.init(
#     # set the wandb project where this run will be logged
#     project="cnn",
#     # track hyperparameters and run metadata
#     config=config
# )

# Load data and model 

