# -*- coding: utf-8 -*-

import csv
import json
import os
import re

import eagerpy as ep
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
from torch.nn.functional import softmax
from PIL import Image
from socket import gethostname


def download_model(model_name):
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True).eval()
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True).eval()
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True).eval()
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True).eval()
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True).eval()
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True).eval()
    elif model_name == "shufflenet_v2_x0_5":
        model = models.shufflenet_v2_x0_5(pretrained=True).eval()
    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(pretrained=True).eval()
    elif model_name == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=True).eval()
    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=True).eval()
    elif model_name == "vgg11":
        model = models.vgg11(pretrained=True).eval()
    elif model_name == "vgg13":
        model = models.vgg13(pretrained=True).eval()
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True).eval()
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=True).eval()
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=True).eval()
    return model


def accuracy(model, input_data, labels):
    examples = len(input_data)
    predictions = model(input_data).argmax(axis=-1)
    return round((predictions == labels).float32().sum().item() / examples * 100, 2) 


def get_effectiveness(model, org_data, modified_data, labels):
    correct_predictions = 0
    effective_attacks = 0
    org_predictions = model(org_data).argmax(axis=-1)
    modified_predictions = model(modified_data).argmax(axis=-1)
    for i, org_prediction in enumerate(org_predictions):
        if (labels[i] == org_prediction).item(): # jesli oryginalna predykcja byla prawidlowa
            correct_predictions += 1
            if not (labels[i] == modified_predictions[i]).item(): # a obecna predykcja jest falszywa
                effective_attacks += 1
    return round(effective_attacks / correct_predictions * 100, 2) 


def download_dataset(base_path, examples, data_format, bounds, dimension):
    images_path = base_path + os.sep + "subset"
    labels_path = base_path + os.sep + "labels.txt"
    images, labels = [], []
    files = os.listdir(images_path)
    files.sort()
    labels_numbers = []

    for idx in range(0, examples):
        # okresl nazwe pliku
        file = files[idx]
        labels_numbers.append(int(re.match(r".+val_0+(.+)\.", file).group(1)))

        # otworz plik
        path = os.path.join(images_path, file)
        image = Image.open(path)
        image = image.resize(dimension)
        image = np.asarray(image, dtype=np.float32)
        
        # jesli obraz jest zapisany w skali szarosci to dodawana jest trzecia os
        if image.ndim == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        # ewentualne przestawienie kanalow
        if data_format == "channels_first":
            image = np.transpose(image, (2, 0, 1))

        # dodanie obrazu do listy obrazow
        images.append(image)

    # pobranie etykiet
    with open(labels_path,"r") as csv_file:
        reader = csv.reader(csv_file)
        for i, line in enumerate(reader, 1):
            if len(labels) == len(files):
                break
            if i in labels_numbers:
                labels.append(int(line[0]))

    # konwersja list do tablic numpy
    images = np.stack(images)
    labels = np.array(labels)

    # wartosci pikseli zdjec domyslnie zawieraja sie w przedziale od 0 do 255
    # jesli ograniczenie wartosci pikseli dla modelu jest inny to nalezy skonwertowac zdjecia
    if bounds != (0, 255):
        images = images / 255 * (bounds[1] - bounds[0]) + bounds[0]
    
    images = ep.from_numpy(ep.torch.zeros(0, device="cpu"), images).raw
    labels = ep.from_numpy(ep.torch.zeros(0, device="cpu"), labels).raw
    return ep.astensor(images), ep.astensor(labels).astype(torch.long)


class FGSM():
    def get_loss_function(self, model, labels):
        def loss_function(input_data):
            logits = model(input_data)
            return ep.crossentropy(logits, labels).sum()
        return loss_function

    def __call__(self, model, input_data, labels, epsilon):
        labels = ep.astensor(labels)
        loss_function = self.get_loss_function(model, labels)
        modified_data = input_data

        # algorytm FGSM
        _, gradients = ep.value_and_grad(loss_function, input_data)
        gradient_sign = gradients.sign()
        modified_data = input_data + epsilon * gradient_sign
        modified_data = ep.clip(modified_data, *model.bounds)
        return modified_data


def save_results(path, epsilons, accuracy, effectiveness, initial_accuracy):
    with open(path + os.sep + "wyniki.csv", mode='w') as csv_file:
        fieldnames = ["epsilon", "dokladnosc_klasyfikacji", "skutecznosc_ataku"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"epsilon": "-",
                         "dokladnosc_klasyfikacji": f"{initial_accuracy}",
                         "skutecznosc_ataku": "-"})
        for i, epsilon in enumerate(epsilons):
            writer.writerow({"epsilon": f"{epsilon}",
                             "dokladnosc_klasyfikacji": f"{accuracy[i]}",
                             "skutecznosc_ataku": f"{effectiveness[i]}"})


def save_modified_images(path, filename, modified_images, eps):
    if type(modified_images) == list:
        for i, images_for_specific_epsilon in enumerate(modified_images):
            save_modified_images(path, filename, images_for_specific_epsilon, eps[i])
    else:
        modified_images = np.transpose(modified_images, axes=(0, 2, 3, 1))
        for i, image in enumerate(modified_images):
            plt.imsave(path + os.sep + f"{filename}_{i+1}_epsilon{eps:0.4f}.jpg", image.numpy())


def save_classes(model, images, labels, destination_path, eps = 0, verbose = False):
    if type(images) == list:
        for i, images_for_specific_epsilon in enumerate(images):
            save_classes(model, images_for_specific_epsilon, labels, destination_path, eps[i])
    else:
        imagenet_classes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "imagenet_classes.json"))
        with open(imagenet_classes_path, "r") as plik_json:
            imagenet_labels = json.load(plik_json)
    
        true_classes = []
        predicted_classes = []
        certainties = []
        for i, _ in enumerate(images):
            if i == 100: #wypisz klasy dla pierwszych 100 zdjec
                break
            image = images[i:i+1]
            predictions = model(image)
            prediction_certainty = softmax(predictions, dim = 1)
            true_classes.append(f"{labels[i].item()}: {imagenet_labels[str(labels[i].item())]}")
            predicted_classes.append(f"{prediction_certainty.argmax().item()}: {imagenet_labels[str(prediction_certainty.argmax().item())]}")
            certainties.append(round(prediction_certainty.max().item(),2))
            if verbose:
                print("\nTOP KLASY")
                print(f"prawdziwa klasa:\t{true_classes[i]}")
                print(f"przewidziana klasa:\t{predicted_classes[i]}")
                print(f"pewnosc:\t\t{certainties[i]} %")
                print("")
    
        with open(destination_path + os.sep + f"klasy_eps{eps:0.4f}" + ".csv", mode='w') as csv_file:
            fieldnames = ["nazwa_pliku", "prawdziwa_klasa", "przewidziana_klasa", "pewnosc"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for i, _ in enumerate(true_classes):
                writer.writerow({"nazwa_pliku": f"ILSVRC2012_val_{i + 1:08d}",
                                 "prawdziwa_klasa": f"{true_classes[i]}",
                                 "przewidziana_klasa": f"{predicted_classes[i]}",
                                 "pewnosc": f"{certainties[i]}"})

def save_time(path, **kwargs):
    host = gethostname()
    with open(path + os.sep + "czasy" + ".txt", mode='w') as file:
        file.write(f"{host}\n\n")
        file.write("czynnosc\t\tczas [s]\n")
        for key, value in kwargs.items():
            file.write(f"{key}\t{value}\n")
