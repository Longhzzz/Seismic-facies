from trainer import trainer
encoder = 'ts_tcc_encoder'
shape = '_w10_s10_attr5_addpos'#_w15_s-10'
param = {
    'train_mode':'ft',
    'device':'cuda:3',
    'batch_size':64,
    'lr':1e-5,
    'rootpath':'',
    'data_name':'boxing_es4cs2',
    'window':10,
    'shift':10,
    # 'propertys':['sx','cdgEnvelope','cdgInsPhs','cdgPhaseShift','cdgQuadraAmp','cdgRmsa',
    #              'cdgSwt','cdgVar'],
    'propertys':['sx','sweetness','instan_amp','90Hz_spectral_decomposition'],
    
    'ssl_method':'ts_tcc',
    'ssl_encoder':encoder,
    'ssl_paraname':'Warp_aug+jitter'+shape,
    'ssl_epoch':60,
    
    'ft_from_mode':'ssl',
    'ft_from_feature':'fe+te',
    'ft_from_ssl_epoch':60,
    'ft_method':'simple_classifier',
    'ft_encoder':encoder,
    'ft_paraname':'simple_freezy'+shape,
    'ft_epoch':200,
    
    'pd_from_mode':'ft',
    'pd_from_feature':'fe+te',
    'pd_from_ft_epoch':100,
    'pd_from_ssl_epoch':60,
    'pd_method':'kmeans',
    'pd_encoder':encoder,
    'pd_paraname':'kmeans'+shape
}

param['ft_from_mode'] = 'ssl'
param['ft_from_feature'] = 'fe+te'
train = trainer(param)
train.finetune()

# param['ft_from_mode'] = 'ssl'
# param['ft_from_feature'] = 'fe'
# train = trainer(param)
# train.finetune()

param['ft_from_mode'] = 'None'
param['ft_from_feature'] = 'fe+te'
train = trainer(param)
train.finetune()

# param['ft_from_mode'] = 'None'
# param['ft_from_feature'] = 'fe'
# train = trainer(param)
# train.finetune()

