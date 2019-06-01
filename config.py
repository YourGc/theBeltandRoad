# coding:utf-8
##  在fold2 256 base的基础上finetune
cfg = {}
cfg['input_size'] = (224,224)
cfg['data_path'] = r'/content/drive/My Drive/train'
cfg['cache_path'] = 'cache'
cfg['mean'] = (0.2,0.2,0.2)
cfg['std'] = (1,1,1)