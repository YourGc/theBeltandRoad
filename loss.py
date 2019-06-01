# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss,self).__init__()

    def forward(self, pred,target):
        '''
        :param pre: logit value output by CNN
        :param target: label
        :return: CELoss
        '''
        #这样写可扩展性好
        print(pred.shape,target.shape)
        print(pred)
        _, preds = torch.argmax(pred, 1)
        print(preds)
        loss = nn.CrossEntropyLoss()(preds, target)
        return loss
