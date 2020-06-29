# -*- coding: utf-8 -*-

import torch
from torchvision import models
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.wide_resnet50_2().to(device)

summary(model, (3, 224, 224))