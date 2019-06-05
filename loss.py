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
        print(target)
        probs = F.softmax(pred,1)
        print(probs[0])
        loss = - (self.alpha) * ( probs ** self.gamma ) * torch.log(1 - probs)
        print(loss[0])
        loss[target] = - (1 - self.alpha) * (probs[target] ** self.gamma) * torch.log(probs)
        print(loss[0])
        exit(0)

        return loss.sum()




