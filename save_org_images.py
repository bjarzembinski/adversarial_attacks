# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np

from utils import download_dataset


def save_org_images(path, filename, images_org):
        images_org = np.transpose(images_org, axes=(0, 2, 3, 1))
        for i, image in enumerate(images_org):
            plt.imsave(path + os.sep + f"{filename}_{i+1}_original.jpg", image.numpy())


data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "imagenet"))
destination_path = data_path + os.sep + "preprocessed"

images, _ = download_dataset(base_path = data_path,
                             examples = 1000,
                             data_format = "channels_first",
                             bounds = (0, 1),
                             dimension = (224, 224))

save_org_images(destination_path, "imagenet", images)
