# coding:utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import tqdm


def eval(model,valloader,criterion,device,cfg):
    model.eval()

    total_loss = 0.0
    bacth_acc_count = 0

    val_size = len(valloader)
    with torch.no_grad():
        for idx,(imgs,labels) in tqdm.tqdm(enumerate(valloader)):

            if device == 'cuda':
                imgs = Variable(imgs.cuda())
                labels = Variable(labels.cuda())
            else:
                imgs, labels = Variable(imgs), Variable(labels)

            output = model(imgs)
            preds = torch.argmax(output,dim=1)
            total_loss += criterion(output,labels)
            bacth_acc_count += torch.sum(preds == labels)

        return bacth_acc_count.float()/(val_size*cfg['batch_size']),total_loss/val_size



