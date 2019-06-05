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
        alpha = 0.1
        gamma = 2

        #数值稳定的softmax
        print(pred)
        out_max,_ = torch.max(pred,0)
        print(out_max,_)
        pred -= out_max
        pred = torch.exp(pred)
        print(pred)
        pros = pred/torch.sum(pred,0)

        print(pros)
        p_out,idx = torch.max(pros,0)
        print(p_out)

        neg_pros = pros[pros == p_out]
        print(neg_pros)
        pos_pros = pros[pros != p_out]
        print(pos_pros)

        neg_loss = -(1-alpha) * (neg_pros ** gamma) * torch.log(1-neg_pros)
        pos_loss = -alpha * ((1 - pos_pros) ** gamma) * torch.log(pos_pros)
        return torch.sum(neg_loss) + torch.sum(pos_loss)



