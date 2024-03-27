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

config={
    "model": "resnet50",
    "epochs": 2,
    "weights": "DEFAULT",
    "batch_size": 32,
    "loss_func": "CrossEntropyLoss",
    "unfrozen_layers": 0
}


wandb.init(
    # set the wandb project where this run will be logged
    project="cnn",
    # track hyperparameters and run metadata
    config=config
)

# Load data and model 
train_dir = 'dataset/training'
valid_dir = 'dataset/validation'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
train_loader, valid_loader, num_classes = load_data(train_dir, 
                                                    valid_dir, 
                                                    batch_size=config["batch_size"], 
                                                    img_size=224) # ResNet50 requires 224x224 images
model = CNN(torchvision.models.resnet50(weights=config["weights"]), num_classes, unfreezed_layers=config["unfrozen_layers"]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
history = model.train_model(train_loader, valid_loader, optimizer, criterion, epochs=config["epochs"])
train_loss = history["train_loss"][-1]
train_acc = history["train_accuracy"][-1]
valid_loss = history["valid_loss"][-1]
valid_acc = history["valid_accuracy"][-1]


wandb.log({"train_acc": train_acc,
           "train_loss": train_loss,
           "valid_acc": valid_acc,
           "valid_loss": valid_loss})
wandb.finish()