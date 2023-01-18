import argparse
import torch
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc

import os
import numpy as np
import matplotlib.pyplot as plt


import MVTec_loader as mvtec

from gradcam import GradCAM

    # for dataloader check: pin pin_memory, batch size 32 in original
mean = 0.
std = 0.
totlen =0
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

num_classes = len(CLASS_NAMES)
for i in range(num_classes):
    class_name = mvtec.CLASS_NAMES[i]   # nuts
    train_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=True, grayscale=False)
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, **kwargs)



    for images, _ in train_loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    totlen += len(train_loader.dataset)
    print(i, mean/totlen)
mean /= totlen
print("mean",mean)