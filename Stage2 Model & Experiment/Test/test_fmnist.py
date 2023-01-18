import argparse
import torch
from torchvision import datasets, transforms

import os
import numpy as np

from model import mnistAE
from sklearn.metrics import roc_curve, roc_auc_score, jaccard_score
import OneClassMnist
from gradcam import GradCAM
import cv2
import fmnist_loader as fmnist
from PIL import Image
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from torchvision import transforms as T

cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')
torch.cuda.set_device(2)
device = torch.device("cuda" if cuda else "cpu")

### Save attention maps  ###
def save_cam(image, filename, gcam):
    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)
    h, w, d = image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    gcam = np.asarray(gcam, dtype=float) + \
        np.asarray(image, dtype=float)
    gcam = 255 * gcam / np.max(gcam)
    gcam = np.uint8(gcam)
    cv2.imwrite(filename, gcam)

def main():
    parser = argparse.ArgumentParser(description='Explainable VAE MNIST Example')
    parser.add_argument('--result_dir', type=str, default='mvtec_test_results', metavar='DIR',
                        help='output directory')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # model options
    parser.add_argument('--latent_size', type=int, default=32, metavar='N',
                        help='latent vector size of encoder')
    parser.add_argument('--model_path', type=str, default='./ckpt/ae_mvtec_model_best', metavar='DIR',
                        help='pretrained model directory')
    parser.add_argument('--one_class', type=int, default=5, metavar='N',
                        help='outlier digit for one-class VAE testing')
    parser.add_argument('--data_path', type=str, default='./data', metavar='DIR',
                        help='directory for storing and finding the dataset')
    parser.add_argument('--layer', type=str, default='encoder.3',
                        help='when true the intersection over union is computed')
    parser.add_argument('--loss', type=str, default='mean',
                        help='when true the intersection over union is computed')
    parser.add_argument('--test', type=str, default='data-mu_mean', 
                        help='when true the intersection over union is computed')
    parser.add_argument('--ssim', default=False, type=str,
                        help='path to latest checkpoint (default: None')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    class_name = fmnist.CLASS_NAMES[args.one_class]
    one_class = args.one_class # Choose the current outlier digit to be 8
    # one_mnist_test_dataset = OneClassMnist.OneMNIST('expVAE/code/data/FashionMNIST', one_class, train=False, transform=transforms.ToTensor())
    mu_sum = torch.zeros(1,1,28,28).to(device)
    train_dataset = fmnist.FmnistDataset(class_name=class_name, is_train=True, grayscale=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    model = mnistAE().to(device)
    checkpoint = torch.load(args.model_path+'_'+'ssim_'+args.layer+'_'+args.loss+'_'+str(args.one_class)+'ssim_'+args.ssim+'.pth')
    model.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        for batch_idx,(data,_)in enumerate(train_loader):
            data = data.to(device)
            mu_sum += data        
        mu_mean = mu_sum/len(train_loader.dataset) 
    test_dataset = fmnist.FmnistDataset(class_name=class_name, is_train=False, grayscale=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    test_steps = len(test_dataset)
    imshape = [1, 28, 28 ]

    # model = ConvVAE(args.latent_size).to(device)
    # model = ConvAutoencoder().to(device)

    gcam = GradCAM(model, target_layer=args.layer, cuda=True) 
    prediction_stack = np.zeros((test_steps, imshape[-1], imshape[-1]), dtype=np.float32)
    gt_mask_stack = np.zeros((test_steps, imshape[-1], imshape[-1]), dtype=np.uint8)
    test_index=0
    for batch_idx, (data, y) in enumerate(test_loader):
        model.eval()
        data = data.to(device)
        #model.zero_grad()
        norm_mu = torch.exp(torch.pow(data-mu_mean,3))
        #norm_mu = data-mu_mean
        normalize  = T.Normalize(mean = norm_mu.mean(), std = norm_mu.std())
        norm_mu = normalize(norm_mu)
        #norm_mu = (norm_mu-norm_mu.min())/(norm_mu.max()-norm_mu.min())
        norm_mu = norm_mu.to(device)
        recon_batch, x = gcam.forward(norm_mu)
        _, s = gcam.forward(norm_mu)
        gcam.backward(s)
        #recon_batch, x = gcam.forward(norm_mu)
        
        #gcam.backward(x)
        gcam_map = gcam.generate() 

        ## Visualize and save attention maps  ##
        if data.size(1) == 1:
            data = data.repeat(1, 3, 1, 1)
        if y.size(1) == 1:
            y1 = y.repeat(1, 3, 1, 1)
        else:
            y1 = y
        for i in range(data.size(0)):
            raw_image = data[i] * 255.0
            ndarr = raw_image.permute(1, 2, 0).cpu().byte().numpy()
            gt = y1[i] *255.0
            gt = gt.permute(1, 2, 0).cpu().byte().numpy()
            # Get the gradcam for this image
            prediction = gcam_map[i].squeeze().cpu().data.numpy()

            # Add prediction and mask to the stacks
            prediction_stack[batch_idx*args.batch_size + i] = prediction
            gt_mask_stack[batch_idx*args.batch_size + i] = y[i]
            gt = Image.fromarray(gt.astype(np.uint8))
            im = Image.fromarray(ndarr.astype(np.uint8))
            im_path = args.result_dir+'_'+args.layer+'_'+args.loss+'_'+args.test+'ssim_'+args.ssim
            if not os.path.exists(im_path):
                os.mkdir(im_path)
            im.save(os.path.join(im_path,
                             "{}-{}-origin.png".format(test_index, str(one_class))))
            gt.save(os.path.join(im_path,
                             "{}-{}-groundtruth.png".format(test_index, str(one_class))))
            file_path = os.path.join(im_path,
                                 "{}-{}-attmap.png".format(test_index, str(one_class)))
            r_im = np.asarray(im)
            save_cam(r_im, file_path, gcam_map[i].squeeze().cpu().data.numpy())
            test_index += 1

    auc = roc_auc_score(gt_mask_stack.flatten(), prediction_stack.flatten())
    print(f"AUROC score: {auc}")

    fpr, tpr, thresholds =  roc_curve(gt_mask_stack.flatten(), prediction_stack.flatten())    
    plt.plot(tpr, fpr, label="ROC")
    plt.xlabel("FPR")            
    plt.ylabel("TPR")
    plt.legend()
    im_path = args.result_dir+'_'+args.layer+'_'+args.loss+'_'+args.test+'ssim_'+args.ssim
    plt.savefig(os.path.join(im_path,"auroc_" + str(args.one_class)+ ".png"))

            # Compute IoU        
    max_val = np.max(prediction_stack)

    max_steps = 100
    best_thres = 0
    best_iou = 0
    # Ge the IoU for 100 different thresholds
    for i in range(1, max_steps):
        thresh = i/max_steps*  max_val
        prediction_bin_stack = prediction_stack > thresh
        iou = jaccard_score(gt_mask_stack.flatten(), prediction_bin_stack.flatten())
        if iou > best_iou:
            best_iou = iou
            best_thres = thresh
    print("Best threshold;", best_thres)
    print("Best IoU score:", best_iou)

    return

if __name__ == '__main__':
    main()