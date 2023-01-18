import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import random_shapes
import os


def img_resize(DST_W,DST_H):
    imglunkuo = np.random.randint(256, size=(4,4,3))
    global_x0 = np.random.randint(14)
    global_y0 = np.random.randint(29)
    global_x1 = global_x0 + DST_W
    global_y1 = global_y0 + DST_H
    #global_x0 = int(((28 -DST_W ) / 2))
    #global_y0 = int(((28 -DST_H ) / 2))
    #global_x1 = int(global_x0 + DST_W)
    #global_y1 = int(global_y0 + DST_H)
    imglunkuo = cv2.resize(imglunkuo, (DST_W, DST_H), interpolation=cv2.INTER_AREA)
    resized1 = cv2.imread("./fmnist/train/1.jpg")
    resized1[global_y0:global_y1, global_x0:global_x1] = imglunkuo  # imglunkuo是小图,resized1是大图，其他参数是左上点和右下点
    cv2.imwrite("./fmnist/train/1_crpt.jpg",resized1)
    resized2 = np.zeros((28,28, 3), np.uint8)
    resized3 = np.zeros((4,4,3), np.uint8)
    resized3.fill(255)
    resized2[global_y0:global_y1, global_x0:global_x1] = resized3
    cv2.imwrite("./fmnist/train/1_gt.jpg",resized2)
    return resized1
#img_resize(4,4)
def random_shape(imgpath,imgname, min_shapes=1, max_shapes=1, shape=None, min_size=3, max_size=8, allow_overlap=True):
    
    result =random_shapes((28, 28), min_shapes=min_shapes, max_shapes=max_shapes, shape=shape, 
                           min_size=min_size, max_size=max_size, allow_overlap=allow_overlap, num_channels=3, multichannel=True)
    resized1 = cv2.imread(imgpath)
    img, labels = result
    img, _ = result
    img[img<255] = 0
    img1 = 255-img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = np.array([-1, -1, -1])
    high = np.array([180, 255, 46])
    dst = cv2.inRange(src=img1, lowerb=low, upperb=high) 
    xy = np.column_stack(np.where(dst==0))
    #print(xy)
    for c in xy:
        #print(c)
        #注意颜色值是(b,g,r)，不是(r,g,b)
        #坐标:c[1]是x,c[0]是y
        x = np.random.randint(256)
        y = np.random.randint(256)
        z = np.random.randint(256)
        cv2.circle(img=resized1, center=(int(c[1]), int(c[0])), radius=1, color=(255, 255, 255),thickness=1)
    resized1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #final = cv2.addWeighted(beta, 0.5,alpha,0.5, 0)
    #print(final)
    #resized1[labels[0][1][0][0]:labels[0][1][0][0],labels[0][1][1][0]:labels[0][1][0][1]] = result[labels[0][1][0][0]:labels[0][1][0][0],labels[0][1][1][0]:labels[0][1][0][1]]
    final = resized1
    if not os.path.exists(os.path.join(os.path.dirname(imgpath),'ground_truth')):
        os.makedirs(os.path.join(os.path.dirname(imgpath),'ground_truth'))
    if not os.path.exists(os.path.join(os.path.dirname(imgpath),'crop')):
        os.makedirs(os.path.join(os.path.dirname(imgpath),'crop'))
    cv2.imwrite(os.path.join(os.path.dirname(imgpath)+'/crop',imgname+'_crop.jpg'),final)
    cv2.imwrite(os.path.join(os.path.dirname(imgpath)+'/ground_truth',imgname+'_gt.jpg'),img)
    return img
#img = random_shape()
 
def getFileList(dir,Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)
    
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            getFileList(newDir, Filelist, ext)
 
    return Filelist
 
org_img_folder='./fmnist/test_images'
 
# 检索文件
imglist = getFileList(org_img_folder, [], 'jpg')
print('本次执行检索到 '+str(len(imglist))+' 张图像\n')
 
for imgpath in imglist:
    imgname= os.path.splitext(os.path.basename(imgpath))[0]
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    random_shape(imgpath,imgname)
    # 对每幅图像执行相关操作

