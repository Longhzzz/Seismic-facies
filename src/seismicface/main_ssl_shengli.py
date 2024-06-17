from trainer import trainer
import gc
encoder = 'transformer_encoder'
epoch = 200
shape = '_w6_s-3_bs_1024_lars_addWNone_gausswindow2_cnn2_attnmask.5_postivepair1'
param = {
    'train_mode':'ssl',
    'device':'cuda:1',
    'batch_size':1024,
    'lr':1e-4,
    'rootpath':'',
    'data_name':'shengli_ngs3',
    'window':6,
    'shift':-3,
    'ext':0,
    't':2,
    'log':'log17_attrchoice_paper_alltools_attnmask_moreepoch',
    'propertys':['sx','cdgInsPhs','cdgGenSpectDecom','cdgDipDev','cdgVar','cdgQuadraAmp','cdgSwt',
        'cdgRmsa','cdgEdge','cdgAmpConst','cdgIsoFreqComp','cdgInsQua','cdgPhaseShift',
        'cdgStrucSmooth','cdgCurv','cdgInsBandW','cdgLoStDip','cdgInsFrq','cdgEnvelope'],
    
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
    
for i in range(0,epoch+1,50):
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

if 'cls' in shape:
    param['train_mode'] = 'predict'
    param['ft_from_mode'] = 'ssl'
    param['ft_from_feature'] = 'fe+te'
    param['pd_from_mode'] = 'ssl'
    param['pd_from_feature'] = 'fe+te'
    param['pd_method'] = 'cls'
    param['pd_from_ssl_epoch'] = 20
    train2 = trainer(param)
    train2.predict()