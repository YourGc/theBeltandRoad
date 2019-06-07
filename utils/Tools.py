# coding:utf-8
import random
import torch.nn as nn
import os
import shutil
import cv2
import numpy as np
random.seed(666)

#模型权重初始化
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

#创建目录
def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

#分离验证集
def split_dataset(cfg):
    #随机分出20%验证集
    SPLIT = 0.1
    train_path = cfg['train_path']
    val_path = cfg['val_path']
    create_dir(val_path)

    labels = os.listdir(train_path)
    for label in labels:
        label_path = os.path.join(train_path,label)
        save_path = os.path.join(val_path,label)
        pic_names = sorted(os.listdir(label_path))
        picked_names = random.sample(pic_names,k=round(SPLIT * len(pic_names)))
        create_dir(save_path)

        for pn in picked_names:
            shutil.move(os.path.join(label_path,pn),os.path.join(save_path,pn))

#获得图像均值和方差
def img_mean_std(cfg):
    train_path = cfg['train_path']
    labels = os.listdir(train_path)
    #input img size
    count = 100*100.0
    #B G R
    means = np.zeros((3,),dtype=np.float128)
    train_count = 0
    for label in labels:
        label_path = os.path.join(train_path, label)
        pic_names = sorted(os.listdir(label_path))
        for name in pic_names:
            img = cv2.imread(os.path.join(label_path,name))
            img = np.array(img)
            img = img.sum(axis=0).sum(axis = 0)
            means += img/count
            train_count +=1

    means = means / train_count
    means /= 255
    print("train set BGR mean is {} , {} , {}".format(means[0],means[1],means[2]))

    #B G R
    stds = np.zeros((3,),dtype=np.float128)
    train_count = 0
    for label in labels:
        label_path = os.path.join(train_path, label)
        pic_names = sorted(os.listdir(label_path))
        for name in pic_names:
            img = cv2.imread(os.path.join(label_path,name))
            img = np.array(img,dtype = np.float)
            img[:,:,0] -= means[0] * 255
            img[:, :, 1] -= means[1] * 255
            img[:, :, 2] -= means[2] * 255
            img = np.square(img)
            img = img.sum(axis=0).sum(axis = 0)
            stds += img / count
            train_count +=1

    stds = stds / train_count
    stds = np.sqrt(stds)
    stds /= 255
    print("train set std is {} , {} , {}".format(stds[0],stds[1],stds[2]))

#直方图均衡化
def Histogram_Equalization(img):
    '''
    :param img: img file(cv2)
    :return: img after HE
    '''
    #split channle
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # merge
    result = cv2.merge((bH, gH, rH))
    return result

if __name__ == '__main__':
    dir = r'F:\Chrome-Download\2019bigdata\2019bigdata\train_image\train\001'
    save_dir = r'F:\Chrome-Download\2019bigdata\2019bigdata\train_image\train\001_aug'
    create_dir(save_dir)
    files = os.listdir(dir)
    for file in files:
        path = os.path.join(dir,file)
        img = cv2.imread(path)
        img = Histogram_Equalization(img)

        cv2.imwrite(os.path.join(save_dir,file),img)







