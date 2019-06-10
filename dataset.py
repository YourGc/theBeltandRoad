# coding:utf-8

from torch.utils.data.dataset import Dataset
from utils.Tools import *
import pickle
import os
import cv2
import numpy as np
import random
import torch
random.seed(666)

class custom_Dataset(Dataset):
    def __init__(self,cfg,phase,**kwargs):
        super(custom_Dataset).__init__()
        self.cache_path = cfg['cache_path']
        self.input_size = cfg['input_size']
        self.mean = cfg['mean']
        self.std = cfg['std']
        self.phase = phase
        self.cfg = cfg
        self.data= self.load_cache(cfg)



    def Histogram_Equalization(self,img):
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

    def load_cache(self,cfg):
        print('load {} cache...'.format(self.phase))
        if os.path.exists(os.path.join(cfg['cache_path'],'cache_{}.pkl'.format(self.phase))):
            with open(os.path.join(cfg['cache_path'],'cache_{}.pkl'.format(self.phase)),'rb') as f:
                data = pickle.load(f)
            f.close()
            return data
        else:
            print('cannot find cache file,making...')
            if not os.path.exists(cfg['cache_path']):
                os.mkdir(cfg['cache_path'])
            data= self.load_data(cfg)
            with open(os.path.join(cfg['cache_path'],'cache_{}.pkl'.format(self.phase)),'wb') as f:
                pickle.dump(data,f)
            f.close()
            data = data_sample(cfg,self.phase)
            return data

    #move load data to Tool.py
    def load_data(self,cfg):
        data={
            'x':[],
            'y':[]
        }
        labels = os.listdir(cfg['{}_path'.format(self.phase)])
        for label in labels:
            root_path = os.path.join(cfg['{}_path'.format(self.phase)],label)
            img_names = os.listdir(root_path)
            for img_name in img_names:
                data['x'].append(os.path.join(root_path,img_name))
                data['y'].append(np.array(int(label) - 1))

        data['x'] = np.array(data['x'])
        data['y'] = np.array(data['y'])
        return data

    def preprocessing(self,img):

        img /=255
        img -= self.mean
        img /= self.std

        return img

    def get_img(self,index):
        img = cv2.imread(self.data['x'][index])
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        #img = self.Histogram_Equalization(img)
        img_array = np.array(img, dtype=np.float32)
        img_array = self.preprocessing(img_array)
        img_array = np.transpose(img_array,(2,0,1))
        return img_array

    def get_visit(self,filename):
        visit_path = self.cfg['cache_path'] + r'/visit_npy' + filename + '.npy'
        visit = np.load(visit_path)
        return visit

    def __getitem__(self,index):
        filename = self.data['x'][index].split('/')[-1].strip('.jpg')
        visit = torch.from_numpy(self.get_visit(filename))
        img = torch.from_numpy(self.get_img(index))
        lable = torch.from_numpy(np.array(self.data['y'][index]))

        return (img,visit),lable

    def __len__(self):
        assert len(self.data['x']) == len(self.data['y']) , "customDataSet Error"
        return len(self.data['x'])
