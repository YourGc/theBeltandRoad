# coding:utf-8

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import tqdm

from torch.utils.data import DataLoader
from senet import se_resnet50
from dataset import custom_Dataset
from config import cfg
from loss import CELoss

def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def train(model,optimizer,scheduler,cfg):
    trainsets = custom_Dataset(cfg)
    trainloader = DataLoader(trainsets, num_workers=4,batch_size=cfg['batch_size'],shuffle=True)

    out_dir = '{}_{}_{}'.format(cfg['model_name'], time.strftime("%Y%m%d"),time.strftime("%H%M%S"))

    criterion = CELoss()
    save_dir = os.path.join(cfg['checkpoint_dir'],out_dir)
    create_dir(save_dir)


    # if cfg['finetune_model'] is not None:
    #     model.load_state_dict(torch.load(cfg['finetune_model']), strict=False)

    #logger = txt_logger(out_dir, 'training', 'log.txt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step = 0
    best_score = 0

    for epoch in range(cfg['epochs']):
        total_loss = 0
        train_iterator = tqdm.tqdm(trainloader, ncols=50)
        for idx, (imgs, labels) in enumerate(train_iterator):
            optimizer.zero_grad()
            labels = labels.to(device)
            imgs = imgs.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            status = '[{0}] lr = {1:.7f} batch_loss = {2:.3f} epoch_loss = {3:.3f} '.format(
                epoch + 1, scheduler.get_lr()[0], loss.data, total_loss / (idx + 1))

            train_iterator.set_description(status)

            if step % cfg['show_freq'] == 0:
                print(status)
            step += 1
        scheduler.step(epoch)
        model_path = os.path.join(out_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
        # if True:  # epoch>20:
        #     print("Evaluate~~~~~")
        #     model.eval()
        #     eval_info = valid_aug(model, evalloader, cfg['img_size'])
        #     eval_score = eval_info['ious']
        #     logger.add_scalar('ious', eval_score, step)
        #     logger.add_scalar('acc', eval_info['acc'], step)
        #     logger.add_scalar('recall', eval_info['recall'], step)
        #     model.train()
        #     if best_score < eval_score:
        #         best_score = eval_score
        #         model_path = os.path.join(out_dir, 'best.pth')
        #         torch.save(model.state_dict(), model_path)
        #     print("mean ap : {:.4f} , best ap: {:.4f}".format(eval_score, best_score))
        #     logger.print_info(epoch)


if __name__ == '__main__':
    create_dir(cfg['checkpoint_dir'])

    model = se_resnet50(9,None)
    optimizer = optim.SGD(model.parameters(), lr=cfg['base_lr'], momentum=0.9, weight_decay=1e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(cfg['epochs'] // 9) + 1)

    #summary(model,(3,224,224))
    dataset = custom_Dataset(cfg)
    train(model,optimizer,scheduler,cfg)

