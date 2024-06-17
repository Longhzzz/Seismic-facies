from trainer import trainer
encoder = 'transformer_encoder'
shape = '_w10_s0_bs_1024_attr10'#_w15_s-10'
param = {
    'train_mode':'predict',
    'device':'cuda:2',
    'batch_size':2048,
    'lr':1e-5,
    'rootpath':'',
    'data_name':'jintao',
    'log':'log6',
    'window':10,
    'shift':0,
    # 'propertys':['sx','cdgEnvelope','cdgInsPhs','cdgPhaseShift','cdgQuadraAmp','cdgRmsa',
    #              'cdgSwt','cdgVar'],
    # 'propertys':['sx','sweetness','instan_amp','90Hz_spectral_decomposition'],
    # 'propertys':['sx','sweetness','instan_amp','lith46'],
    'propertys':['sx','coherent_energy','crossline_dip','glcm_homogeneity','inline_dip','k_curvedness',
                 'k_s_index','peak_freq_cmp','peak_mag_cmp','sobel_filter_similarity'],
    # 'propertys':['sx','coherent_energy','k_curvedness','peak_freq_cmp','peak_mag_cmp',],
    
    'ssl_method':'ts_tcc_channelT',
    'ssl_encoder':encoder,
    'ssl_paraname':'Mask+jitter_aug'+shape,
    'ssl_epoch':60,
    
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
    'pd_from_ssl_epoch':60,
    'pd_method':'classifier',#kmeans when pd_from_mode = None or ssl,classifier when pe_from_mode = finetune
    'pd_encoder':encoder,
    'pd_paraname':'simple_freezy'+shape
}


# # param['data_name'] = 'boxing_es4cs4'
# param['ft_from_mode'] = 'None'
# param['ft_from_feature'] = 'fe+te'
# param['pd_from_mode'] = 'ft'
# param['pd_from_feature'] = 'fe'
# param['pd_method'] = 'classifier'
# train = trainer(param)
# train.predict()


# # param['data_name'] = 'boxing_es4cs4'
# param['ft_from_mode'] = 'ssl'
# param['ft_from_feature'] = 'fe+te'
# param['pd_from_mode'] = 'ft'
# param['pd_from_feature'] = 'fe'
# param['pd_method'] = 'classifier'
# train = trainer(param)
# train.predict()

# param['data_name'] = 'boxing_es4cs4'
param['ft_from_mode'] = 'ssl'
param['ft_from_feature'] = 'fe+te'
param['pd_from_mode'] = 'ssl'
param['pd_from_feature'] = 'fe+te'
param['pd_method'] = 'kmeans'
train = trainer(param)
train.predict()


# # param['data_name'] = 'boxing_es4cs4'
param['ft_from_mode'] = 'None'
param['ft_from_feature'] = 'fe'
param['pd_from_mode'] = 'None'
param['pd_from_feature'] = 'fe'
param['pd_method'] = 'kmeans'
train = trainer(param)
train.predict()




