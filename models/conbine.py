# coding:utf-8
from models.senet import se_resnet50_no_Linear
import torch.nn as nn
import torch

class ConbinedSE50(nn.Module):
    def __init__(self,num_classes,dropout = None):
        super(ConbinedSE50,self).__init__()
        self.image_net = se_resnet50_no_Linear()
        self.visit_net = se_resnet50_no_Linear()
        self.fc = nn.Linear(in_features=2048,out_features=num_classes)
        self.dropout = nn.Dropout(p=dropout) if dropout != None else None

    def forward(self, imgs,visits):
        x_img = self.image_net(imgs)
        x_visit = self.visit_net(visits)
        x = x_img + x_visit
        if self.dropout != None:
            x = self.dropout(x)
        x = self.fc(x)
        return x
