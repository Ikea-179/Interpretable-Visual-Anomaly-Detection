from sklearn.cluster import KMeans
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from torchvision.models import resnet50
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import argparse
import torch
import BTAD_loader as btad
import torchvision
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
torch.cuda.set_device(1)

def calc_metrics(testy, scores):
    precision, recall, _ = precision_recall_curve(testy, scores)
    roc_auc = roc_auc_score(testy, scores)
    prc_auc = auc(recall, precision)

    return roc_auc, prc_auc

def main():
    parser = argparse.ArgumentParser(description='Explainable VAE MNIST Example')
    parser.add_argument('--result_dir', type=str, default='./train_results_btad', metavar='DIR',
                        help='output directory')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt', metavar='DIR',
                        help='ckpt directory')
    parser.add_argument('--data_path', type=str, default='./data', metavar='DIR',
                        help='directory for storing and finding the dataset')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--TRAIN_RAND_SEED', default=42, type=int, help='random seed used selecting training data')
    parser.add_argument('--TEST_RAND_SEED', default=[42, 89, 2, 156, 491, 32, 67, 341, 100, 279], type=list,
                        help='random seed used selecting test data.'
                             'The number of elements should equal to "test_rep_count" for reproductivity of validation.'
                             'When the length of this list is less than "test_rep_count", seed is randomly generated')
    parser.add_argument('--test_rep_count', default=10, type=int,
                        help='counts of test repeats per one trained model. For a model, test data selection and evaluation are repeated.')
    # model options
    parser.add_argument('--one_class', type=int, default=3, metavar='N',
                        help='inlier digit for one-class VAE training')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    TRAIN_RAND_SEED = args.TRAIN_RAND_SEED
    TEST_RAND_SEED = args.TEST_RAND_SEED
    test_rep_count = args.test_rep_count
    kwargs = {}
    one_class = args.one_class # Choose the inlier digit to be 3
    # one_mnist_train_dataset = OneClassMnist.OneMNIST('expVAE/code/data/FashionMNIST', one_class, train=True, download=False, transform=transforms.ToTensor())
    # one_mnist_test_dataset = OneClassMnist.OneMNIST('expVAE/code/data/FashionMNIST', one_class, train=False, transform=transforms.ToTensor())
    class_name = btad.CLASS_NAMES[args.one_class]
    train_dataset = btad.BTADDataset(class_name=class_name, is_train=True, grayscale=False, root_path=args.data_path)
    print(len(train_dataset))
    test_dataset = btad.BTADDataset(class_name=class_name, is_train=False, grayscale=False, root_path=args.data_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, **kwargs)
    pr_scores = []
    roc_scores = []
    nus = [200,300,400]
    # train model and evaluate with changing parameter nu
    # train with nu
    for batch_idx, (data, gt,label) in enumerate(train_loader):
        X_train = data.view(len(train_dataset),-1).detach().numpy()
        Y_train = label
    for batch_idx, (data, gt,label) in enumerate(test_loader):
        X_test = data.view(len(test_dataset),-1).detach().numpy()
        Y_test = label
        # Apply standard scaler to output from resnet50


    for nu in nus:
        total_pr = 0
        total_roc = 0

        # repeat test by randomly selected data and evaluate
        for j in range(test_rep_count):
            # select test data and test
            if j < len(TEST_RAND_SEED):
                TEST_SEED = np.random.RandomState(TEST_RAND_SEED[j])
            else:
                TEST_SEED = np.random.RandomState(np.random.randint(0, 10000))
            Y_pred = KMeans(n_clusters=2, init='k-means++',max_iter=nu,random_state=9).fit_predict(X_test)
            # calculate evaluation metrics
            roc_auc, prc_auc = calc_metrics(Y_test, Y_pred)

            total_pr += prc_auc
            total_roc += roc_auc

        # calculate average
        total_pr /= test_rep_count
        total_roc /= test_rep_count

        pr_scores.append(total_pr)
        roc_scores.append(total_roc)

        print('--- nu : ', nu, ' ---')
        print('PR AUC : ', total_pr)
        print('ROC_AUC : ', total_roc)

    print('***' * 5)
    print('PR_AUC MAX : ', max(pr_scores))
    print('ROC_AUC MAX : ', max(roc_scores))
    print('ROC_MAX_NU : ', nus[int(np.argmax(roc_scores))])

if __name__ == '__main__':
    main()