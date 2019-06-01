# coding:utf-8

import torch
from torchsummary import  summary
from senet import se_resnet50
from dataset import custom_Dataset
from config import cfg

model = se_resnet50(9,None)
#summary(model,(3,224,224))
dataset = custom_Dataset(cfg)


