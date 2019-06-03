# coding:utf-8
##  在fold2 256 base的基础上finetune
cfg = {}


#model config
cfg['model_name'] = r'se_resneXt'
#数据参数
cfg['input_size'] = (224,224)
cfg['train_path'] = r'train'
#cfg['data_path'] = r'/content/drive/My Drive/train'
cfg['val_path'] = r'val'
cfg['cache_path'] = 'cache'
#BGR
cfg['mean'] = (0.62138,0.53756,0.46789)
cfg['std'] = (0.14795,0.16502,0.18105)

#训练相关
cfg['batch_size'] = 64
cfg['base_lr'] = 0.01
cfg['epochs'] = 100
cfg['print_freq'] = 10
cfg['checkpoint_dir'] = r'output'
#cfg['checkpoint_dir'] = r'/content/drive/My Drive/output'
cfg['checkpoint_freq'] = 10
