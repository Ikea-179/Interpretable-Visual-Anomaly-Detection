import torch
import cv2
import numpy as np
import torchvision
'''mnist_train = torchvision.datasets.FashionMNIST(root=r'./data/fmnist/',
                    train=True,  download=True)
mnist_test = torchvision.datasets.FashionMNIST(root=r'./data/fmnist/',
                    train=False,  download=True)'''
train_data=torch.load("./data/fmnist/FashionMNIST/processed/training.pt")[0].numpy()
test_data=torch.load("./data//fmnist/FashionMNIST/processed/test.pt")[0].numpy()
print(train_data.shape)
print(test_data.shape)
#print(test_data[0])
for i  in range(train_data.shape[0]):     
    cv2.imwrite("./fmnist/train/"+str(i)+".jpg",train_data[i][:,:,np.newaxis])
for i  in range(test_data.shape[0]):     
    cv2.imwrite("./fmnist/test/"+str(i)+".jpg",test_data[i][:,:,np.newaxis])


import numpy as np
import cv2
import os

def save_mnist_to_jpg(mnist_image_file, mnist_label_file, save_dir):
    if 'train' in os.path.basename(mnist_image_file):
        num_file = 60000
        prefix = 'train'
    else:
        num_file = 10000
        prefix = 'test'
    with open(mnist_image_file, 'rb') as f1:
        image_file = f1.read()
    with open(mnist_label_file, 'rb') as f2:
        label_file = f2.read()
    image_file = image_file[16:]
    label_file = label_file[8:]
    for i in range(0,10):
        class_dir = save_dir+'/'+str(i)+'/'
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    for i in range(num_file):
        label = int(label_file[i])
        image_list = [int(item) for item in image_file[i*784:i*784+784]]
        image_np = np.array(image_list, dtype=np.uint8).reshape(28,28,1)
        save_name = os.path.join(save_dir+'/'+str(label)+'/', '{}_{}_{}.jpg'.format(prefix, i, label))
        cv2.imwrite(save_name, image_np)
        print ('{} ==> {}_{}_{}.jpg'.format(i, prefix, i, label))

if __name__ == '__main__':
    train_image_file = './data/fmnist/FashionMNIST/raw/train-images-idx3-ubyte'
    train_label_file = './data/fmnist/FashionMNIST/raw/train-labels-idx1-ubyte'
    test_image_file = './data/fmnist/FashionMNIST/raw/t10k-images-idx3-ubyte'
    test_label_file = './data/fmnist/FashionMNIST/raw/t10k-labels-idx1-ubyte'

    save_train_dir = './fmnist/train_images/'
    save_test_dir ='./fmnist/test_images/'

    if not os.path.exists(save_train_dir):
        os.makedirs(save_train_dir)
    if not os.path.exists(save_test_dir):
        os.makedirs(save_test_dir)

    save_mnist_to_jpg(train_image_file, train_label_file, save_train_dir)
    save_mnist_to_jpg(test_image_file, test_label_file, save_test_dir)