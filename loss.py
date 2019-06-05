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
        print(pred[0])
        out_max,_ = torch.max(pred,1)
        print(out_max[0],_[0])
        pred -= out_max
        pred = torch.exp(pred)
        print(pred[0])
        pros = pred/torch.sum(pred,0)

        print(pros[0])
        p_out,idx = torch.max(pros,0)
        print(p_out[0])

        neg_pros = pros[pros == p_out]
        print(neg_pros[0])
        pos_pros = pros[pros != p_out]
        print(pos_pros[0])

        neg_loss = -(1-alpha) * (neg_pros ** gamma) * torch.log(1-neg_pros)
        pos_loss = -alpha * ((1 - pos_pros) ** gamma) * torch.log(pos_pros)
        return torch.sum(neg_loss) + torch.sum(pos_loss)



