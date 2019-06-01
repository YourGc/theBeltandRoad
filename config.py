# coding:utf-8
##  在fold2 256 base的基础上finetune
cfg = {}
cfg['input_size'] = (3,224,224)
cfg['data_path'] = r'F:\Chrome-Download\2019bigdata\2019bigdata\train_image\train'
cfg['cache_path'] = 'cache'
cfg['mean'] = ()
cfg['std'] = ()