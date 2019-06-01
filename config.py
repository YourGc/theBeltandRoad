# coding:utf-8
##  在fold2 256 base的基础上finetune
cfg = {}


#model config
cfg['model_name'] = r'se_resneXt'
#数据参数
cfg['input_size'] = (224,224)
cfg['data_path'] = r'/content/drive/My Drive/train'
cfg['cache_path'] = 'cache'
cfg['mean'] = (0.2,0.2,0.2)
cfg['std'] = (1,1,1)

#训练相关
cfg['batch_size'] = 16
cfg['base_lr'] = 0.01
cfg['epochs'] = 100
cfg['show_freq'] = 10
cfg['checkpoint_dir'] = '/content/drive/My Drive/output'
cfg['checkpoint_frequency'] = 10