# coding:utf-8
import random
import os
import shutil
random.seed(666)


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def split_dataset(cfg):
    #随机分出20%验证集
    train_path = cfg['train_path']
    val_path = cfg['val_path']
    create_dir(val_path)

    labels = os.listdir(train_path)
    for label in labels:
        label_path = os.path.join(train_path,label)
        save_path = os.path.join(val_path,label)
        pic_names = sorted(os.listdir(label_path))
        picked_names = random.sample(pic_names,k=round(0.2 * len(pic_names)))
        create_dir(save_path)

        for pn in picked_names:
            shutil.move(os.path.join(label_path,pn),os.path.join(save_path,pn))



