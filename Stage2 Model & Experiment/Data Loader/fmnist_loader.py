import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import save_image

CLASS_NAMES = ['0','1','2','3','4','5','6','7','8','9']


class FmnistDataset(Dataset):
    def __init__(self, root_path='./fmnist', class_name='0', is_train=True,
                 resize=28, cropsize=28, grayscale=True):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.root_path = root_path
        self.class_name = class_name
        print("For class",class_name)
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.fmnist_folder_path = root_path

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # Of just Hazlenut
        self.mean = np.array([0.2397, 0.1764, 0.1709])
        self.std = np.array([0.1650, 0.0728, 0.0414])

        # Own computed
        # self.mean = np.array([0.4305, 0.3999, 0.3900])
        # self.std = np.array([0.1822, 0.1733, 0.1624])

        # from the loader found online
        # self.mean = np.array([0.485, 0.456, 0.406])
        # self.std = np.array([0.229, 0.224, 0.225])
        # set transforms
        if grayscale is True:
            self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                T.Grayscale(num_output_channels=1),
                                T.CenterCrop(cropsize),
                                T.ToTensor()
                                # T.Normalize((0.5), (0.5))
                                ])
        else:
            self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                        T.CenterCrop(cropsize),
                                        T.ToTensor()])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

        # Random flips and random rotations
        self.augmentation_x = T.Compose([ T.ToPILImage(),
                                          T.RandomRotation(np.pi/6),
                                          T.RandomHorizontalFlip(),
                                          T.RandomVerticalFlip(),
                                          T.ToTensor()])
        self.inv_normalize = T.Normalize(mean = -self.mean/self.std, std = 1/self.std)
        self.normalize = T.Normalize(mean = self.mean, std = self.std)

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x)#.convert('RGB')
        x = self.transform_x(x)

        if self.is_train == True:
            x = self.augmentation_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
            # print("y is 0 , mask is",y)
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        #x = self.normalize(x)

        # return x, y, mask
        return x, mask

    def __len__(self):
        return len(self.x)

    def unnormalize(self, x):

        unnorm = self.inv_normalize(x)
        return unnorm
    def load_dataset_folder(self):
        phase = 'train_images' if self.is_train else 'test_images'
        x, y, mask,label = [], [], [],[]

        img_dir = os.path.join(self.fmnist_folder_path, phase, self.class_name)
        gt_dir = os.path.join(self.fmnist_folder_path,phase, self.class_name, 'ground_truth')

        # load images
        if phase == 'train_images':
            img_fpath_list = sorted([os.path.join(img_dir, f)
                                        for f in os.listdir(img_dir)
                                        if f.endswith('.jpg')])
        else:
            img_fpath_list = sorted([os.path.join(img_dir,'crop', f)
                                        for f in os.listdir(img_dir+'/crop/')
                                        if f.endswith('.jpg')])
            gt_fpath_list = sorted([os.path.join(gt_dir, f)
                                        for f in os.listdir(gt_dir)
                                        if f.endswith('.jpg')]) 
        x.extend(img_fpath_list)
        # load gt labels
        if phase == 'train_images':
            y.extend([0] * len(img_fpath_list))
            mask.extend([None] * len(img_fpath_list))

        else:
            y.extend([1] * len(img_fpath_list))
            img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
            mask.extend(gt_fpath_list)
        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)