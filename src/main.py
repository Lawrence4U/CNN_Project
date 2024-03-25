import os

current_directory = os.getcwd()
os.chdir(current_directory)
print("Ubicación actual:", current_directory)

from cnn import CNN
import torchvision
from torchvision import transforms
from cnn import load_data
from cnn import load_model_weights
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import json

def get_label(imagen):
    # def imshow(inp, title=None):
    #     """Imshow for Tensor."""
    #     inp = inp.numpy().transpose((1, 2, 0))
    #     inp = np.clip(inp, 0, 1)
    #     plt.imshow(inp)
    #     if title is not None:
    #         plt.title(title)
    #     plt.pause(0.001)  # pause a bit so that plots are updated


    # Pytorch has many pre-trained models that can be used for transfer learning
    classification_models = torchvision.models.list_models(module=torchvision.models)
    print(classification_models)


    # Load data and model 
    train_dir = '../dataset/training'
    valid_dir = '../dataset/validation'
    train_loader, valid_loader, num_classes = load_data(train_dir, 
                                                        valid_dir, 
                                                        batch_size=32, 
                                                        img_size=224) # ResNet50 requires 224x224 images
    model = CNN(torchvision.models.resnet50(weights='DEFAULT'), num_classes)




    model_weights = torch.load('modelo.pth')
    my_trained_model = CNN(torchvision.models.resnet50(weights='DEFAULT'), num_classes)
    my_trained_model.load_state_dict(model_weights)

    # Cargar una imagen
    # imagen = Image.open("torre.jpg")

    # Preprocesar la imagen
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagen = transform(imagen)
    imagen = imagen.unsqueeze(0)  # Añadir dimensión del lote

    # Pasar la imagen por el modelo
    output = my_trained_model(imagen)
    # Obtener las etiquetas predichas
    _, predicted = torch.max(output, 1)

    # Mostrar las etiquetas predichas
    classes = [predicted.item()]

    out = torchvision.utils.make_grid(output)
    classnames = train_loader.dataset.classes
    for x in classes:
        title = classnames[x]
    print(title)
    # imshow(out, title=title)
    return title
