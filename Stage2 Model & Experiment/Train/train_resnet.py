import argparse

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn as nn
import cv2
import pytorch_ssim

import shutil
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
#from gradcam import GradCAM
from PIL import Image
from model import ConvVAE, ConvAutoencoder, ConvAE_mvtec,ResNet,ResidualBlock
import OneClassMnist
import MVTec_loader as mvtec
import matplotlib.pyplot as plt
cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')

device = torch.device("cuda" if cuda else "cpu")
torch.cuda.set_device(1)

# def loss_function(recon_x, x, mu, logvar):
#     # reconstruction loss
#     BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

#     # KL divergence loss
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return BCE + KLD

def loss_function(recon_x, x, mu, gcam):
    B = recon_x.shape[0]
    loss = F.binary_cross_entropy(recon_x.view(B, -1), x.view(B, -1))
    cam_loss = gcam.mean()-0.5*gcam.std()
    return loss + cam_loss*100000000

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x,y,z):
        loss = F.cross_entropy(y,z)+x.mean()#-0.05*x.std()
        return (loss)

criterion = My_loss()

torch.autograd.set_detect_anomaly(True)

def pca_error(x,s, matrix):
    ss = torch.matmul(s,matrix.T)
    a = torch.norm(x-ss,p=1)
    return a

def proj_error(matrix):
    a = torch.square(torch.norm(torch.matmul(matrix, torch.transpose(matrix)-torch.eye(32).to(device),p=1)))
    return a


def save_cam(image, gcam):
    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)
    h, w, d = image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    gcam = np.asarray(gcam, dtype=float) + \
        np.asarray(image, dtype=float)
    gcam = 255 * gcam / np.max(gcam)
    gcam = np.uint8(gcam)
    return gcam


### Training #####
def train(epoch, model, train_loader, optimizerR,  args, schedulerR, schedulerG,  ssim):
    model.train()
    train_loss = 0
    cam = GradCAM(model, target_layers=args.layer,use_cuda=True)
    for batch_idx, (data, _,label) in enumerate(train_loader):
        data = data.to(device)
        #label = torch.Tensor(label.float())
        label = label.to(device)
        grayscale_cam = cam(input_tensor=data, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        cam_array = []
        img_array = []
        for j in range(len(data)):            
            img = data[j].cpu().numpy()
            visualization = show_cam_on_image(img.transpose((1, 2, 0)), grayscale_cam, use_rgb=False)
            cam_array.append(visualization)
        img = torch.tensor(cam_array).float()
        images = model(data)
        loss = criterion(img,images,label)
        optimizerR.zero_grad()
        loss.backward()
        optimizerR.step()
        #schedulerR.step()
        train_loss += loss.item()


    train_loss /= len(train_loader.dataset)/args.batch_size

    return train_loss

### Validating ####
def test(epoch, model, test_loader, args, ssim):
    model.eval()
    test_loss = 0
    pred = 0
    predict = []
    label = []
    i = 0
    correct = 0
    total = 0 
    for batch_idx, (data,_, labels) in enumerate(test_loader):
        data = data.to(device)
        labels = labels.to(device)
        label.append(labels)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        predict.append(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    test_loss = 100 * correct / total

    return test_loss


def save_checkpoint(state, is_best, outdir,args):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    checkpoint_file = os.path.join(outdir, 'ae_mvtec_checkpoint'+'_'+args.name_layer+'_'+args.loss+'_'+str(args.one_class)+'ssim_'+args.ssim+'.pth')
    best_file = os.path.join(outdir, 'ae_mvtec_model_best_ssim'+'_'+args.name_layer+'_'+args.loss+'_'+str(args.one_class)+'ssim_'+args.ssim+'.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)

def main():
    model =  ResNet(ResidualBlock, [2, 2, 2]).to(device)
    model.train()
    optimizerR = optim.Adam(model.parameters(), lr=1e-2)
    parser = argparse.ArgumentParser(description='Explainable VAE MNIST Example')
    parser.add_argument('--result_dir', type=str, default='./train_results_mvtec', metavar='DIR',
                        help='output directory')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt', metavar='DIR',
                        help='ckpt directory')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--data_path', type=str, default='./data', metavar='DIR',
                        help='directory for storing and finding the dataset')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--resume', default=False, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None')

    # model options
    parser.add_argument('--latent_size', type=int, default=32, metavar='N',
                        help='latent vector size of encoder')
    parser.add_argument('--one_class', type=int, default=3, metavar='N',
                        help='inlier digit for one-class VAE training')
    parser.add_argument('--layer', type=list, default= [model.layer3[-1]],
                        help='when true the intersection over union is computed')
    parser.add_argument('--name_layer', type=str, default= 'model.layer3[-1]',
                        help='when true the intersection over union is computed')
    parser.add_argument('--loss', type=str, default='mean',
                        help='when true the intersection over union is computed')
    parser.add_argument('--ssim', default=False, type=str,
                        help='path to latest checkpoint (default: None')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    one_class = args.one_class # Choose the inlier digit to be 3
    # one_mnist_train_dataset = OneClassMnist.OneMNIST('expVAE/code/data/FashionMNIST', one_class, train=True, download=False, transform=transforms.ToTensor())
    # one_mnist_test_dataset = OneClassMnist.OneMNIST('expVAE/code/data/FashionMNIST', one_class, train=False, transform=transforms.ToTensor())
    class_name = mvtec.CLASS_NAMES[args.one_class]
    train_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=True, grayscale=False, root_path=args.data_path)
    print(len(train_dataset))
    test_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=False, grayscale=False, root_path=args.data_path)
    
    # kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_dataset = torch.utils.data.ConcatDataset([train_dataset,test_dataset])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    # train_loader = torch.utils.data.DataLoader( one_mnist_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # test_loader = torch.utils.data.DataLoader( one_mnist_test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # model = ConvVAE(args.latent_size).to(device)
    # model = ConvAutoencoder().to(device)
    
    # mean = 0.
    # std = 0.
    # for images, _ in train_loader:
    #     batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
    #     images = images.view(batch_samples, images.size(1), -1)
    #     print(torch.min(images[0]))
    #     print(torch.max(images[0]))

    # mean /= len(train_loader.dataset)
    # std /= len(train_loader.dataset)

    # print(mean)
    # print(std)

    
    optimizerG = optim.Adam(model.parameters(), lr=1e-3)
    schedulerR = torch.optim.lr_scheduler.OneCycleLR(optimizerR, max_lr=0.1, div_factor = 100, total_steps=args.epochs*args.batch_size)
    schedulerG = torch.optim.lr_scheduler.OneCycleLR(optimizerG, max_lr=0.1, div_factor = 100, total_steps=args.epochs*args.batch_size)
    lrs = []
    steps = []
    start_epoch = 0
    best_test_loss = np.finfo('f').min

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume+'_'+args.name_layer+'_'+args.loss+'_'+str(args.one_class)+'ssim_'+args.ssim+'.pth'):
            print('=> loading checkpoint %s' % args.resume+'_'+args.name_layer+'_'+args.loss+'_'+str(args.one_class)+'ssim_'+args.ssim+'.pth')
            checkpoint = torch.load(args.resume+'_'+args.name_layer+'_'+args.loss+'_'+str(args.one_class)+'ssim_'+args.ssim+'.pth')
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizerR.load_state_dict(checkpoint['optimizerR'])
            print('=> loaded checkpoint %s' % args.resume)
        else:
            print('=> no checkpoint found at %s' % args.resume)

    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_loss= train(epoch, model, train_loader, optimizerR,  args, schedulerR, schedulerG,  args.ssim)
        # print(scheduler.get_last_lr())
        # lrs.append(scheduler.get_last_lr()[0])
        steps.append(epoch)
        with torch.no_grad():
            test_loss = test(epoch, model, test_loader,args, args.ssim) 
   
        print('Epoch [%d/%d] loss: %.3f val_accuracy: %.3f' % (epoch + 1, args.epochs, train_loss, test_loss))

        is_best = test_loss > best_test_loss
        best_test_loss = max(test_loss, best_test_loss)
        save_checkpoint({
            'epoch': epoch,
            'best_test_loss': best_test_loss,
            'state_dict': model.state_dict(),
            'optimizerR': optimizerR.state_dict(),
        }, is_best, os.path.join('./',args.ckpt_dir),args)
        for batch_idx, (data, _,label) in enumerate(test_loader):
            img = data[0].view(256,256,3).cpu().numpy()
            print(img.size())
            input_tensor = preprocess_image(img,
                                            mean=[0.2397, 0.1764, 0.1709],
                                            std=[0.1650, 0.0728, 0.0414])
            cam = GradCAM(model, target_layers=args.layer)
            targets = []

            grayscale_cam = cam(input_tensor=input_tensor.cuda(), targets=None)

            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            img = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            save_dir = os.path.join('./',args.result_dir+'_'+args.name_layer+'_'+args.loss+'_'+str(args.one_class)+'ssim_'+args.ssim)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_image(data[0].view(1, 3, 256, 256), os.path.join(save_dir, str(epoch) +'_origin_sample' + '.png'))
            cv2.imwrite( os.path.join(save_dir, str(epoch) + '_gcam_sample' + '.png'), img)
        # Visualize sample validation result
       # with torch.no_grad():
            # sample = torch.randn(8, 100, 1, 1).to(device)
            # sample = model.decoder(sample).cpu()


       
    # plt.figure()
    # plt.plot(steps, lrs, label='OneCycle')
    # plt.savefig('./lr.jpg')


if __name__ == '__main__':
    main()