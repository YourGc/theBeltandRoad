# coding:utf-8
from models.senet import se_resnet50_no_Linear
from torchsummary import  summary
model = se_resnet50_no_Linear(isVisit=True)
summary(model,(7,26,24))
