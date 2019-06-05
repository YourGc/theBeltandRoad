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

        #softmax
        print('---pred')
        print(pred.shape,pred[0])
        pros = F.softmax(pred,dim=1)
        p_max,idx = torch.max(pros,1)
        print("--pros--")
        print(pros.shape,pros[0])
        print("--pmax--")
        print(p_max.shape,p_max[0])

        p_max = torch.unsqueeze(p_max,1)
        print(p_max)

        neg_pros = pros[pros == p_max]
        print(neg_pros)
        #print(neg_pros[0])
        pos_pros = pros[pros != p_max]
        print(pos_pros)
        #print(pos_pros[0])
        exit(0)
        neg_loss = -(1-alpha) * (neg_pros ** gamma) * torch.log(1-neg_pros)
        pos_loss = -alpha * ((1 - pos_pros) ** gamma) * torch.log(pos_pros)
        return torch.sum(neg_loss) + torch.sum(pos_loss)



