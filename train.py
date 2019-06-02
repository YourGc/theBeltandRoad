# coding:utf-8


import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import tqdm

from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from senet import se_resnet50
from dataset import custom_Dataset
from config import cfg
from loss import CELoss
from eval import eval
from utils.Tools import *



def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train(model,optimizer,scheduler,cfg):
    trainsets = custom_Dataset(cfg,phase = 'train')
    trainloader = DataLoader(trainsets, num_workers=4,batch_size=cfg['batch_size'],shuffle=True)

    valsets = custom_Dataset(cfg, phase='val')
    valloader = DataLoader(valsets, num_workers=4, batch_size=cfg['batch_size'], shuffle=True)
    print(len(valloader))

    out_dir = '{}_{}_{}'.format(cfg['model_name'], time.strftime("%Y%m%d"),time.strftime("%H%M%S"))
    criterion = CELoss()
    save_dir = os.path.join(cfg['checkpoint_dir'],out_dir)

    writer = SummaryWriter('run')
    model_path = os.path.join(save_dir, '{}_epoch{}.pth')
    create_dir(save_dir)

    model.train(True)
    model.apply(weight_init)
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    # if cfg['finetune_model'] is not None:
    #     model.load_state_dict(torch.load(cfg['finetune_model']), strict=False)

    #logger = txt_logger(out_dir, 'training', 'log.txt')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    step = 0
    best_score = 0
    min_loss = 100.0

    tic_batch = time.time()
    for epoch in range(cfg['epochs']):
        train_loss = 0.0
        for idx, (imgs, labels) in enumerate(trainloader):
            #print(imgs.shape)
            optimizer.zero_grad()
            if device == 'cuda':
                imgs = Variable(imgs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(imgs), Variable(labels)
            preds = model(imgs)
            loss = criterion(preds, labels)
            output = torch.argmax(preds,dim=1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            #print(output,labels)
            cur_correctrs = torch.sum(output == labels.data)
            batch_acc = (cur_correctrs.float()) / (len(output))

            if step % cfg['print_freq'] == 0:
                print('[Epoch {}/{}]-[batch:{}/{}]  Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                      epoch+1, cfg['epochs'], idx + 1, len(trainloader), loss.item(), batch_acc, \
                    cfg['print_freq']/(time.time()-tic_batch)))
                tic_batch = time.time()
            step += 1

        writer.add_scalar('Train',train_loss,epoch+1)
        train_loss = 0.0
        if epoch !=0 and epoch % cfg['checkpoint_freq'] == 0:
            torch.save(model.state_dict(), model_path.format(cfg['model_name'],epoch))

        if True:  # epoch>20:
            print("Evaluate at epoch {}".format(epoch + 1))
            model.eval()
            eval_acc,eval_loss = eval(model,valloader,criterion,device)
            writer.add_scalar('Val', eval_loss, epoch + 1)
            model.train()
            if best_score < eval_acc:
                best_score = eval_acc
                best_model_path = os.path.join(save_dir, 'best.pth')
                torch.save(model.state_dict(), best_model_path)
            if min_loss > eval_loss:
                min_loss = eval_loss

            print("Epoch {} : Accu {:.4f} , best Accu: {:.4f} --- mean Loss {:.6f} , min Loss {:.6f}".format(epoch,eval_acc, best_score,eval_loss,min_loss))

        scheduler.step(eval_loss)

if __name__ == '__main__':
    create_dir(cfg['checkpoint_dir'])

    if not os.path.exists(cfg['val_path']):
        split_dataset(cfg)
    model = se_resnet50(9,None)
    optimizer = optim.SGD(model.parameters(), lr=cfg['base_lr'], momentum=0.9, weight_decay=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=3,verbose=True,)

    #summary(model,(3,224,224))

    train(model,optimizer,scheduler,cfg)

