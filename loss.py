# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self,weight = None):
        super(CELoss,self).__init__()
        self.gamma = 2
        self.alpha = 0.1
        self.nll_loss = torch.nn.NLLLoss(weight)
    def forward(self, pred,target):
        '''
        :param pre: logit value output by CNN
        :param target: label
        :return: CELoss
        '''
        probs = F.softmax(pred,1)
        return self.nll_loss(self.alpha * (1 - probs) ** self.gamma * probs, target)




