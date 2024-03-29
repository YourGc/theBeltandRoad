# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, y_pred,y_true):
        loss = nn.CrossEntropyLoss()(y_pred,y_true)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, class_num=9, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        pos_idx = class_mask.gt(0.5)
        neg_idx = class_mask.le(0.5)
        #print(pos_idx.shape,neg_idx.shape)
        # #neg loss
        # neg_class_mask  = 1 - class_mask

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        # neg loss
        neg_probs = P[neg_idx] #(P * neg_class_mask).sum(1).view(-1,1)
        probs = P[pos_idx]#(P * class_mask).sum(1).view(-1, 1)
        # print(probs.shape,neg_probs.shape)

        eps = 1e-8
        neg_log_p = (1 - neg_probs + eps).log()
        log_p = ( probs + eps ).log()
        # print('neg_log_p size= {}'.format(neg_log_p.size()))
        # print('log_p size= {}'.format(log_p.size()))
        # print(neg_probs)

        #alpha = 1
        neg_batch_loss = - (torch.pow(neg_probs, self.gamma)) * neg_log_p
        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        # print(batch_loss.shape,neg_batch_loss.shape)
        # print('-----bacth_loss------')
        # print("pos_loss:{}".format(batch_loss.mean().item()))
        # print("neg_loss:{}".format(neg_batch_loss.mean().item()))

        loss = neg_batch_loss.mean() + batch_loss.mean()
        # if self.size_average:
        #     loss = loss.mean()
        # else:
        #     loss = loss.sum()
        return loss




