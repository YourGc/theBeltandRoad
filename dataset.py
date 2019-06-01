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
        if not os.path.join(cfg['cache_path']):
            os.mkdir(cfg['cache_path'])

        self.cache_path = cfg['cache_path']
        self.data= self.load_cache(self.cache_path,cfg)

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

    def load_cache(self,path,cfg):
        print('load cache...')
        if os.path.exists(path):
            with open(os.path.join(path,'cache_train.pkl'),'rb') as f:
                data = pickle.load(f)
            f.close()
            return data
        else:
            print('cannot find cache file,making...')
            data= self.load_data(cfg)
            index = [i for i in range(len(self.data['x']))]
            random.shuffle(index)
            #shuffle
            data['x'] = data['x'][index]
            data['y'] = data['y'][index]
            with open(os.path.join(path,'cache_train.pkl'),'w') as f:
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
            root_path = os.path.join(cfg['data_path'],label)
            img_names = os.listdir(root_path)
            for img_name in img_names:
                img = cv2.imread(os.path.join(root_path,img_name))
                img = cv2.resize(img,cfg['input_size'],interpolation=cv2.INTER_LINEAR)
                img = self.Histogram_Equalization(img)
                img = self.preprocessing(img,cfg)
                img_array = np.array(img)
                data['x'].append(img_array)
                data['y'].append(label)

        data['x'] = np.array(data['x'])
        data['y'] = np.array(data['y'])
        return data

    def preprocessing(self,img,cfg):
        img /=255
        img -= cfg['mean']
        img /= cfg['std']
        return img

    def __getitem__(self,index):
        return self.data['x'][index],self.data['y'][index]

    def __len__(self):
        assert len(self.data['x']) == len(self.data['y']) , "customDataSet Error"
        return len(self.data)
