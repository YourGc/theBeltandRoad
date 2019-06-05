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
        print(pred[0])
        print(target[0])
        probs = F.softmax(pred,1)
        print(probs[0])
        print('compute')
        print((probs ** self.gamma * self.alpha) [0] )
        print(torch.log(1-probs)[0])
        #torch.log是以e为底数的ln
        loss = - (self.alpha) * ( probs ** self.gamma ) * torch.log(1 - probs)
        print(loss[0])
        print(target.shape)
        print(probs[target].shape,probs[target][0])
        loss[target] = - (1 - self.alpha) * (probs[target] ** self.gamma) * torch.log(probs)
        print(loss[0])
        exit(0)

        return loss.sum()




