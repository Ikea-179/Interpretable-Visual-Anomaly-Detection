import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.measure  
from PIL import Image  
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus
import cv2
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2397, 0.1764, 0.1709],
                             std=[0.1650, 0.0728, 0.0414])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2397, 0.1764, 0.1709],
                             std=[0.1650, 0.0728, 0.0414])
    ])
}


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(65536, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model1 = ResNet(ResidualBlock, [2, 2, 2])

from torch.nn import functional as F
num_epochs = 20
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(2)
model1 = model1.to(device)
# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

data_path_train = "hazelnut/train"
train_dataset = torchvision.datasets.ImageFolder(root=data_path_train,transform=transforms.ToTensor())
data_path_test = "hazelnut/test"
test_dataset = torchvision.datasets.ImageFolder(root=data_path_test,transform=transforms.ToTensor())
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=16, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=16, 
                                          shuffle=False)


# Loss and optimizer
optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x,y,z):
        loss = F.cross_entropy(y,z)+x.mean()#-0.5*x.std()
        return (loss)
criterion = My_loss()

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        target_layers = [model1.layer3[-1]]
        cam = GradCAM(model1, target_layers=target_layers)
        targets = []
        grayscale_cam = cam(input_tensor=images, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        cam_array = []
        img_array = []
        
        for j in range(len(images)):            
            img = images[j].cpu().numpy()
            visualization = show_cam_on_image(img.transpose((1, 2, 0)), grayscale_cam, use_rgb=False)
            #img1 = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            #cam_array.append(img1.transpose((2, 0, 1)))
            cam_array.append(visualization)
            '''img2 = img.transpose((1, 2, 0))*255
            img2 = img2.astype(np.uint8)
            img2 = cv2.applyColorMap(img2, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
            img2 = img2.astype(float)
            img_array.append(img2.transpose((2, 0, 1)))'''
        #k1 = 0.05
        img = torch.tensor(cam_array).float()
        #images = torch.tensor(img_array)
        # Forward pass
        images = model1(images)
        loss = criterion(img,images,labels)
        #loss.requires_grad = True
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    torch.save(model1.state_dict(), 'Gradcam20std0mean1.ckpt')
    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
