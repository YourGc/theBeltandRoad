# coding:utf-8

from torch.utils.data.dataset import Dataset
import pickle
import os
import cv2
import numpy as np
import random
random.seed(666666)

class custom_Dataset(Dataset):
    def __init__(self,cfg):
        super(custom_Dataset).__init__()
        self.cache_path = cfg['cache_path']
        self.input_size = cfg['input_size']
        self.mean = cfg['mean']
        self.std = cfg['std']
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
        print('load cache...')
        if os.path.exists(os.path.join(cfg['cache_path'],'cache_train.pkl')):
            with open(os.path.join(cfg['cache_path'],'cache_train.pkl'),'rb') as f:
                data = pickle.load(f)
            f.close()
            return data
        else:
            print('cannot find cache file,making...')
            if not os.path.exists(cfg['cache_path']):
                os.mkdir(cfg['cache_path'])
            data= self.load_data(cfg)
            #人工shuffle Dataload有shuffle接口
            # index = [i for i in range(len(data['x']))]
            # random.shuffle(index)
            # #shuffle
            # data['x'] = data['x'][index]
            # data['y'] = data['y'][index]
            with open(os.path.join(cfg['cache_path'],'cache_train.pkl'),'wb') as f:
                pickle.dump(data,f)
            f.close()
            return data

    def load_data(self,cfg):
        data={
            'x':[],
            'y':[]
        }
        labels = os.listdir(cfg['data_path'])
        for label in labels:
            print(label)
            root_path = os.path.join(cfg['data_path'],label)
            img_names = os.listdir(root_path)
            for img_name in img_names:
                data['x'].append(os.path.join(root_path,img_name))
                data['y'].append(label)

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
        img = self.Histogram_Equalization(img)
        img_array = np.array(img, dtype=np.float32)
        img_array = self.preprocessing(img_array)
        return img_array

    def __getitem__(self,index):
        img = self.get_img(index)
        lable = self.data['y'][index]
        return img,lable

    def __len__(self):
        assert len(self.data['x']) == len(self.data['y']) , "customDataSet Error"
        return len(self.data)
