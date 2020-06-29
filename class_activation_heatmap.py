# -*- coding: utf-8 -*-

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch import topk
from utils import download_model
import numpy as np
import skimage.transform
import os


def save_heatmaps_for_model(model_name, base_path, examples = 100):
    for i in range(1, examples):
        photo_number = i
    
        # zapis oryginalnych zdjec
        path = (os.path.join(base_path,
                             "imagenet",
                             "preprocessed",
                             f"imagenet_{photo_number}_original.jpg"))
        destination_path = (os.path.join(base_path,
                                         "WYNIKI",
                                         "mapy_ciepla",
                                         f"{model_name}",
                                         f"imagenet_{photo_number}.jpg"))
        save_heatmap(model_name, path, destination_path)
    
        # zapis zdjec sfabrykowanych
        path = (os.path.join(base_path,
                             "WYNIKI",
                             "1",
                             f"{model_name}",
                             "data",
                             f"imagenet_{photo_number}_epsilon0.0100.jpg"))
        destination_path = (os.path.join(base_path,
                                       "WYNIKI",
                                       "mapy_ciepla",
                                       f"{model_name}",
                                       f"imagenet_{photo_number}_adv.jpg"))
        save_heatmap(model_name, path, destination_path)


def save_heatmap(model_name, path, destination_path):
    image = Image.open(path)
    
    # klasa potrzebana do ustawienia hooka na ostatnia warstwe konwolucjna
    class Hook():
        features = None
        def __init__(self, layer):
            self.hook = layer.register_forward_hook(self.get_activation_map)
        def get_activation_map(self, module, input, output):
            self.activation_maps = ((output.cpu()).data).numpy()
        def remove(self):
            self.hook.remove()

    # inicjalizacja modelu
    model = download_model(model_name)
    if "resnet" in model_name:
        last_convolutional_layer_name = list(model._modules.keys())[-3]
        classification_layer_name = list(model._modules.keys())[-1]
    else:
        last_convolutional_layer_name = list(model._modules.keys())[-2]
        classification_layer_name = list(model._modules.keys())[-1]
    
    # definicja normalizacji do sredniej i odchylenia standardowego zbioru ImageNet
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # definicja preprocessingu - zmiana rozmiaru, konwersja na tensor i normalizacja
    preprocessing = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalization])
    
    # transformacja potrzebna do wyswietlenia zdjecia
    reshaping = transforms.Compose([transforms.Resize((224,224))])
    
    # preprocessing zdjecia
    tensor = preprocessing(image)
    
    # ustawienie hooka na ostatnia warstwe konwolucyjna
    last_convolutional_layer = model._modules.get(last_convolutional_layer_name)
    hook = Hook(last_convolutional_layer)
    
    # podanie zdjecia na wejscie modelu
    predictions = model(Variable((tensor.unsqueeze(0)), requires_grad=True))
    hook.remove()
    
    # podanie predykcji na funkcję softmax w celu uzyskania prawdopodobienstw
    probabilities = softmax(predictions).data.squeeze()
    
    # pobranie parametrow warstwy klasyfikacyjnej
    classification_layer = list(model._modules.get(classification_layer_name).parameters())
    classification_layer = np.squeeze(classification_layer[0].cpu().data.numpy())
    
    # okreslenie klasy o najwyzszyej pewnosci predykcji
    predicted_class = topk(probabilities,1)[1].int()
    
    # stworzenie mapy ciepla aktywacji klas
    heatmap = get_heatmap(hook.activation_maps, classification_layer, predicted_class)
    
    # zapis zdjecia z nalozona mapa ciepla aktywacji klas
    plt.imshow(reshaping(image))
    plt.imshow(skimage.transform.resize(heatmap[0], tensor.shape[1:3]), alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.savefig(destination_path, bbox_inches='tight')


# funkcja tworzaca mape ciepla
def get_heatmap(activation_maps, classification_layer_weights, class_idx):
    _, nc, h, w = activation_maps.shape
    # iloczyn skalarny teensorow
    heatmap = classification_layer_weights[class_idx].dot(activation_maps.reshape((nc, h*w)))
    heatmap = heatmap.reshape(h, w)
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / np.max(heatmap)
    return [heatmap]

model_name = "shufflenet_v2_x0_5"
base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
save_heatmaps_for_model(model_name, base_path)
