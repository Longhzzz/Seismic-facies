from trainer import trainer
import gc
import random
import numpy as np
import torch
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
encoder = 'transformer_encoder'
epoch = 500
shape = '_w8_s2_bs_1024_lars_addWNone_gausswindow1.5_cnn_attnmask.5_postivepair1_ext1_t2'
param = {
    'train_mode':'ssl',
    'device':'cuda:4',
    'batch_size':1024,
    'lr':1e-4,
    'rootpath':'',
    'data_name':'jintao',
    'window':8,
    'shift':2,
    'ext':1,
    't':2,
    'log':'log17_attrchoice_paper_alltools_attnmask_moreepoch',
    'propertys':['sx','coherent_energy','crossline_dip','glcm_homogeneity','inline_dip','k_curvedness',
                 'k_s_index','peak_freq_cmp','peak_mag_cmp'],
    
    'ssl_method':'ts_tcc_channelT',
    'ssl_encoder':encoder,
    'ssl_paraname':'attr_choice_aug'+shape,
    'ssl_epoch':epoch,
    
    'ft_from_mode':'ssl',
    'ft_from_feature':'fe+te',
    'ft_from_ssl_epoch':60,
    'ft_method':'simple_classifier',
    'ft_encoder':encoder,
    'ft_paraname':'simple_freezy'+shape,#注意修改！
    'ft_epoch':200,
    'n_class':4,
    
    'pd_from_mode':'ft',#ssl,ft,None
    'pd_from_feature':'fe',#only when pd_from_mode=ssl, it work, usually with kmeans
    'pd_from_ft_epoch':200,
    'pd_from_ssl_epoch':200,
    'pd_method':'classifier',#kmeans when pd_from_mode = None or ssl,classifier when pe_from_mode = finetune
    'pd_encoder':encoder,
    'pd_paraname':'simple_freezy'+shape
}

traine = trainer(param)
traine.ssl_train()

del traine
gc.collect()
    
for i in range(500,-1,-50):
    param['train_mode'] = 'predict'
    param['ft_from_mode'] = 'ssl'
    param['ft_from_feature'] = 'fe+te'
    param['pd_from_mode'] = 'ssl'
    param['pd_from_feature'] = 'fe+te'
    param['pd_method'] = 'kmeans'
    # param['propertys'] = ['sx']
    # param['pd_paraname'] = 'simple_freezy_sx'+shape
    param['pd_from_ssl_epoch'] = i
    train2 = trainer(param)
    train2.predict()
    del train2
    gc.collect()

# param['train_mode'] = 'predict'
# param['ft_from_mode'] = 'ssl'
# param['ft_from_feature'] = 'fe+te'
# param['pd_from_mode'] = 'ssl'
# param['pd_from_feature'] = 'fe+te'
# param['pd_method'] = 'kmeans'
# # param['propertys'] = ['sx']
# # param['pd_paraname'] = 'simple_freezy_sx'+shape
# param['pd_from_ssl_epoch'] = 90
# train2 = trainer(param)
# train2.predict()

# param['train_mode'] = 'predict'
# param['ft_from_mode'] = 'ssl'
# param['ft_from_feature'] = 'fe+te'
# param['pd_from_mode'] = 'ssl'
# param['pd_from_feature'] = 'fe+te'
# param['pd_method'] = 'kmeans'
# # param['propertys'] = ['sx']
# # param['pd_paraname'] = 'simple_freezy_sx'+shape
# param['pd_from_ssl_epoch'] = 80
# train2 = trainer(param)
# train2.predict()

# param['train_mode'] = 'predict'
# param['ft_from_mode'] = 'ssl'
# param['ft_from_feature'] = 'fe+te'
# param['pd_from_mode'] = 'ssl'
# param['pd_from_feature'] = 'fe+te'
# param['pd_method'] = 'kmeans'
# # param['propertys'] = ['sx']
# # param['pd_paraname'] = 'simple_freezy_sx'+shape
# param['pd_from_ssl_epoch'] = 60
# train2 = trainer(param)
# train2.predict()

# param['train_mode'] = 'predict'
# param['ft_from_mode'] = 'ssl'
# param['ft_from_feature'] = 'fe+te'
# param['pd_from_mode'] = 'ssl'
# param['pd_from_feature'] = 'fe+te'
# param['pd_method'] = 'kmeans'
# # param['propertys'] = ['sx']
# # param['pd_paraname'] = 'simple_freezy_sx'+shape
# param['pd_from_ssl_epoch'] = 40
# train2 = trainer(param)
# train2.predict()

# param['train_mode'] = 'predict'
# param['ft_from_mode'] = 'ssl'
# param['ft_from_feature'] = 'fe+te'
# param['pd_from_mode'] = 'ssl'
# param['pd_from_feature'] = 'fe+te'
# param['pd_method'] = 'kmeans'
# # param['propertys'] = ['sx']
# # param['pd_paraname'] = 'simple_freezy_sx'+shape
# param['pd_from_ssl_epoch'] = 20
# train2 = trainer(param)
# train2.predict()

# param['train_mode'] = 'predict'
# param['ft_from_mode'] = 'ssl'
# param['ft_from_feature'] = 'fe+te'
# param['pd_from_mode'] = 'ssl'
# param['pd_from_feature'] = 'fe+te'
# param['pd_method'] = 'kmeans'
# # param['propertys'] = ['sx']
# # param['pd_paraname'] = 'simple_freezy_sx'+shape
# param['pd_from_ssl_epoch'] = 0
# train2 = trainer(param)
# train2.predict()

# param['train_mode'] = 'predict'
# param['ft_from_mode'] = 'None'
# param['ft_from_feature'] = 'fe'
# param['pd_from_mode'] = 'None'
# param['pd_from_feature'] = 'fe'
# param['pd_method'] = 'kmeans'
# train3 = trainer(param)
# train3.predict()

# if 'cls' in shape:
#     param['train_mode'] = 'predict'
#     param['ft_from_mode'] = 'ssl'
#     param['ft_from_feature'] = 'fe+te'
#     param['pd_from_mode'] = 'ssl'
#     param['pd_from_feature'] = 'fe+te'
#     param['pd_method'] = 'cls'
#     param['pd_from_ssl_epoch'] = 20
#     train2 = trainer(param)
#     train2.predict()