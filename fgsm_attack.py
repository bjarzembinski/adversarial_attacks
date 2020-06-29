# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path
from foolbox import PyTorchModel
from utils import (
    FGSM, accuracy, get_effectiveness, download_dataset, download_model,
    save_time, save_classes, save_results, save_modified_images)

model_name = "shufflenet_v2_x0_5"
examples = 1000
epsilons = [x/10000 for x in range(1,10)] + [x/1000 for x in range(1,11)] + [0.1]
save_results_flag = True
save_modified_images_flag = True
save_classes_flag = True
data_dirname = "imagenet"
results_destination_dirname = model_name
images_destination_dirname = "data"

data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), data_dirname))
results_destination_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), results_destination_dirname))
images_destination_path = os.path.join(results_destination_path, images_destination_dirname)
Path(images_destination_path).mkdir(parents=True, exist_ok=True)

def initialize_model():
    model = download_model(model_name)
    # normalizacja do sredniej i odchylenia standardowego zbioru ImageNet
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    return PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

if __name__ == "__main__":
    model = initialize_model()
    
    # sciagnij dataset i przetestuj model
    images, labels = download_dataset(base_path = data_path,
                                        examples = examples,
                                        data_format = model.data_format,
                                        bounds = model.bounds,
                                        dimension = (224, 224))

    # wyniki dla niezmodyfikowanych danych
    initial_accuracy = accuracy(model, images, labels)

    # przeprowadz ataki dla kazdego epsilona i oblicz czas ataku
    atak = FGSM()
    duration = {}
    start_attack_time = time.time()
    modified_images = []
    for epsilon in epsilons:
        modified_images.append(atak(model, images, labels, epsilon))
    end_attack_time = time.time()
    duration["czas_ataku__fgsm"] = round(end_attack_time - start_attack_time, 0)
    print("Atak zakonczony")

    # oblicz i zapisz dokladnosc po ataku
    if save_results_flag:
        start_save_time = time.time()
        classification_accuracy = []
        effectiveness = []
        for modified_images_for_one_epsilon in modified_images:
            classification_accuracy.append(accuracy(model, modified_images_for_one_epsilon, labels))
            effectiveness.append(get_effectiveness(model, images, modified_images_for_one_epsilon, labels))
        save_results(results_destination_path, epsilons, classification_accuracy, effectiveness, initial_accuracy)
        end_save_time = time.time()
        duration["czas_zapisu_wynikow"] = round(end_save_time - start_save_time, 0)
        print("Zapisywanie wynikow zakonczone")

    # zapisanie zmodyfikowanych obrazow (tylko dla najwiekszego epsilona)
    if save_modified_images_flag:
        start_save_time = time.time()
        save_modified_images(images_destination_path, data_dirname, modified_images[-1], epsilons[-1])
        end_save_time = time.time()
        duration["czas_zapisu_obrazow"] = round(end_save_time - start_save_time, 0)
        print("Zapisywanie zmodyfikowanych obrazow zakonczone")
    
    # zapisanie klas predykowanych dla konkretnych obrazkow
    if save_classes_flag:
        start_save_time = time.time()
        save_classes(model, images, labels, results_destination_path)
        save_classes(model, modified_images, labels, results_destination_path, epsilons)
        end_save_time = time.time()
        duration["czas_zapisu_klas"] = round(end_save_time - start_save_time, 0)

    # zapisanie czasow
    save_time(results_destination_path, **duration)
