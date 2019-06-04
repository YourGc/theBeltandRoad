# coding:utf-8
import torch.nn as nn
import torch
import argparse
import os
import cv2
import numpy as np
import tqdm
import pandas as pd

from torch.utils.data import DataLoader
from torch.autograd import  Variable
from torch.utils.data.dataset import Dataset
from config import cfg
from senet import se_resnet50

class testDataset(Dataset):
    def __init__(self,args,cfg):
        super(Dataset,self).__init__()
        self.mean = cfg['mean']
        self.std = cfg['std']
        self.input_size = cfg['input_size']
        self.path = args.test_path
        self.data = self.get_data()

    def get_data(self):
        data = []
        img_names = os.listdir(self.path)
        for name in img_names:
            data.append(name)
        return data

    def get_img(self,index):
        img = cv2.imread(os.path.join(self.path,self.data[index]))
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = np.array(img,dtype=np.float32)/255
        img -=self.mean
        img /= self.std
        img = np.transpose(img,(2,0,1))
        return img

    def __getitem__(self, index):
        return self.get_img(index),self.data[index]

    def __len__(self):
        return len(self.data)

def get_args():
    parses  = argparse.ArgumentParser(description = 'Infer config')
    parses.add_argument('--gpus' ,type=str,default='0,1,2,3')
    parses.add_argument('--model_path',type=str,default='best.pth')
    parses.add_argument('--test_path',type=str,default='test')
    args = parses.parse_args()
    return args

def infer():
    args = get_args()
    model = se_resnet50(9,None)
    #model = nn.DataParallel(model,device_ids=[int(i) for i in args.gpus.split(',')])

    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    model.eval()

    testdata = testDataset(args,cfg)
    testloader = DataLoader(testdata,batch_size=1,num_workers=2)

    ans = []
    with torch.no_grad():
        for idx,(img,name) in tqdm.tqdm(enumerate(testloader)):
            img = Variable(img.cuda())
            output = model(img)
            pred = torch.argmax(output).item()
            ans.append([str(name[0]).strip('.jpg'),"00"+str(pred)])

    result = pd.DataFrame(ans,columns=['AreaID,CategoryID'])
    result.sort_values(by=['AreaID'])
    result.to_csv('submit.csv',index=False,header=False,sep = '\t')

if __name__ == '__main__':
    infer()













