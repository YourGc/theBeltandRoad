# coding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import argparse
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

def get_args():
    parses  = argparse.ArgumentParser(description = 'Train config')
    #parses.add_argument('--gpus' ,type=str,default='0,1,2,3')
    parses.add_argument('--model_path',type=str,default=None)
    parses.add_argument('--epoch',type=str,default=None)
    args = parses.parse_args()
    return args

def train(model,optimizer,scheduler,cfg,args):
    trainsets = custom_Dataset(cfg,phase = 'train')
    trainloader = DataLoader(trainsets, num_workers=4,batch_size=cfg['batch_size'],shuffle=True)

    valsets = custom_Dataset(cfg, phase='val')
    valloader = DataLoader(valsets, num_workers=4, batch_size=cfg['batch_size'], shuffle=True)

    out_dir = '{}_{}_{}'.format(cfg['model_name'], time.strftime("%Y%m%d"),time.strftime("%H%M%S"))
    criterion = CELoss()
    save_dir = os.path.join(cfg['checkpoint_dir'],out_dir)

    writer = SummaryWriter(save_dir)
    model_path = os.path.join(save_dir, '{}_epoch{}.pth')
    create_dir(save_dir)

    model.train(True)
    model.apply(weight_init)
    model.cuda()

    #断点重训
    if args.model_path != None:
        if args.epoch == None:
            print('input epoch !')
            exit(0)
        model.load_state_dict(torch.load(args.model_path))
        new_lr = cfg['base_lr'] * (cfg['gamma'] ** int(args.epoch))
        print(new_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    #model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])


    device = "cuda" if torch.cuda.is_available() else "cpu"
    step = 0
    best_score = 0
    best_epoch = -1
    min_loss = 100.0

    tic_batch = time.time()
    for epoch in range(cfg['epochs']):
        train_loss = 0.0
        train_acc = 0.0
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
            train_acc += cur_correctrs.float()
            batch_acc = (cur_correctrs.float()) / (len(output))

            if idx % cfg['print_freq'] == 0:
                print('[Epoch {}/{}]-[batch:{}/{}] lr={:.6f} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                      epoch+1, cfg['epochs'], idx + 1, len(trainloader), scheduler.get_lr()[0],loss.item(), batch_acc, \
                    cfg['print_freq']/(time.time()-tic_batch)))
                tic_batch = time.time()

        train_loss /= len(trainloader)
        train_acc /= len(trainloader) * cfg['batch_size']
        print("Train Epoch {} : mean Accu {:.4f} --- mean Loss {:.6f}".format(epoch + 1,train_acc, train_loss))

        if epoch !=0 and epoch % cfg['checkpoint_freq'] == 0:
            torch.save(model.state_dict(), model_path.format(cfg['model_name'],epoch))

        if True:  # eval:
            print("Evaluate at epoch {}".format(epoch + 1))
            model.eval()
            eval_acc,eval_loss = eval(model,valloader,criterion,device,cfg)
            model.train()
            if best_score < eval_acc:
                best_score = eval_acc
            if min_loss > eval_loss:
                best_model_path = os.path.join(save_dir, 'best.pth')
                torch.save(model.state_dict(), best_model_path)
                best_epoch = epoch
                min_loss = eval_loss

            print("Val Epoch {} : Accu {:.4f} , best Accu: {:.4f} --- mean Loss {:.6f} , min Loss {:.6f} , best at epoch {}".format(epoch + 1,eval_acc, best_score,eval_loss,min_loss,best_epoch))

        writer.add_scalars('loss', {
            'train': train_loss,
            'val': eval_loss
        }, epoch + 1)
        writer.add_scalars('accu', {
            'train': train_acc,
            'val': eval_acc
        }, epoch + 1)
        train_loss = 0.0
        train_acc = 0.0
        scheduler.step()

if __name__ == '__main__':
    create_dir(cfg['checkpoint_dir'])

    if not os.path.exists(cfg['val_path']):
        split_dataset(cfg)

    #img_mean_std(cfg)
    args = get_args()

    model = se_resnet50(9,None)
    optimizer = optim.SGD(model.parameters(), lr=cfg['base_lr'], momentum=0.9, weight_decay=1e-2)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg['gamma'])
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=3,verbose=True,)

    #summary(model,(3,224,224))

    train(model,optimizer,scheduler,cfg,args)

