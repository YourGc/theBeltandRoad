# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self,weight = None):
        super(CELoss,self).__init__()
        self.gamma = 2
        self.alpha = 0.1
    def forward(self, pred,target):
        '''
        :param pre: logit value output by CNN
        :param target: label
        :return: CELoss
        '''
        probs = F.softmax(pred,1)
        loss = - (1 - self.alpha) * ( probs ** self.gamma ) * torch.log(1 - probs)
        loss[target] = - (self.alpha) * (probs[target] ** self.gamma) * torch.log(probs)

        return loss.sum()




