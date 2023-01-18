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

URL = 'https://avires.dimi.uniud.it/papers/btad/btad.zip'
CLASS_NAMES = ['00','01','02','03']


class BTADDataset(Dataset):
    def __init__(self, root_path='./data', class_name='01', is_train=True,
                 resize=256, cropsize=256, grayscale=False):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.root_path = root_path
        self.class_name = class_name
        print("For class",class_name)
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.btad_folder_path = os.path.join(root_path, 'BTech_Dataset_transformed')

        # download dataset if not exist
        # self.download()

        # load dataset
        self.x, self.y, self.mask,self.label = self.load_dataset_folder()

        # Of just 01
        #self.mean = np.array([0.3019, 0.6180, 0.5369])
        #self.std = np.array([0.3161, 0.4495, 0.2392])
        # Of just 02
        self.mean = np.array([0.7777, 0.5985, 0.5381])
        self.std = np.array([0.0782, 0.1152, 0.0707])
        # Of just 03
        # self.mean = np.array([0.3649, 0.5999, 0.6867])
        # self.std = np.array([0.3237, 0.2946, 0.2982])

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
        x, y, mask,label = self.x[idx], self.y[idx], self.mask[idx],self.label[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if self.is_train == True:
            x = self.augmentation_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
            # print("y is 0 , mask is",y)
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        # x = self.normalize(x)

        # return x, y, mask
        return x, mask,label

    def __len__(self):
        return len(self.x)

    def unnormalize(self, x):

        unnorm = self.inv_normalize(x)
        return unnorm



    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask,label = [], [], [],[]

        img_dir = os.path.join(self.btad_folder_path, self.class_name, phase)
        gt_dir = os.path.join(self.btad_folder_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.bmp') or f.endswith('.png')] )
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'ok':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
                label.extend([0] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                if self.class_name == '03':
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.bmp')
                                        for img_fname in img_fname_list]
                else: 
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                        for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)
                label.extend([1] * len(img_fpath_list))
        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask),list(label)

    # def download(self):
    #     """Download dataset if not exist"""

    #     if not os.path.exists(self.mvtec_folder_path):
    #         tar_file_path = './data/BTech_Dataset_transformed'
    #         if not os.path.exists(tar_file_path):
    #             download_url(URL, './data/btad.zip')
    #         print('unzip downloaded dataset: %s' % tar_file_path)
    #         tar = tarfile.open('./data/btad.zip', 'r:zip')
    #         tar.extractall(self.mvtec_folder_path)
    #         tar.close()

    #     return


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
