from ssl_algorithms import *
from tqdm import tqdm
from utils import *
from dataloader import *
import collections
from predictor import *
from ft_algorithms import *
from encoder_list import *
import itertools
from d2l import torch as d2l
from scipy.stats import kurtosis, skew
import copy
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)

def encoder_dict(encoder_name,input_channel,ext,seqlen): 
    if encoder_name == 'ts_tcc_encoder':
        return ts_tcc_encoder(input_channel*ext,seqlen)
    if encoder_name == 'simmtm_encoder':
        return simmtm_encoder(input_channel*ext,seqlen)
    if encoder_name == 'baseline_encoder':
        return baseline_encoder(input_channel*ext,seqlen)
    if encoder_name == 'cnn1d_encoder':
        return cnn1d_encoder(input_channel*ext,seqlen)
    if encoder_name == 'transformer_encoder':
        return transformer_encoder(input_channel,seqlen,ext)
    if encoder_name == 'cnn2d_transformer_encoder':
        return cnn2d_transformer_encoder(input_channel,seqlen,ext)
    if encoder_name == 'transformerDouble_encoder':
        return transformerDouble_encoder(input_channel,seqlen)
    if encoder_name == 'transformerT_encoder':
        return transformerT_encoder(input_channel,seqlen,ext)
    if encoder_name == 'Mid_transformer_encoder':
        return Mid_transformer_encoder(input_channel*ext,seqlen)
    if encoder_name == 'cnn_transformer_encoder':
        return cnn_transformer_encoder(input_channel*ext,seqlen)
def algorithm_dict(name,param): 
    if name == 'ts_tcc':
        return ts_tcc(param['head_dim'],param['encoder'],param['device'],param['batchsize'], param['lr'])
    elif name == 'ts_tcc_ext':
        return ts_tcc_ext(param['head_dim'],param['encoder'],param['device'],param['batchsize'], param['lr'])
    elif name == 'ts_tcc_channelT':
        return ts_tcc_channelT(param['head_dim'],param['encoder'],param['shape'],param['device'],param['batchsize'], param['lr'])
    elif name == 'ts_tcc_mask':
        return ts_tcc_mask(param['head_dim'],param['encoder'],param['shape'],param['device'],param['batchsize'], param['lr'])
    elif name == 'mask_channel_independent':
        return mask_channel_independent(param['head_dim'],param['input_channel'],param['encoder'],param['seq_len'],param['device'], param['lr'])
    elif name == 'simmtm_channel_independent':
        return simmtm_channel_independent(param['head_dim'],param['input_channel'],param['encoder'],param['seq_len'],param['positive_num'],param['device'],param['batchsize'], param['lr'])
    elif name == 'pmsn':
        return pmsn(param['head_dim'],param['encoder'],param['shape'],param['device'],param['batchsize'], param['lr'])
class trainer(object):
    def __init__(self, param):
        self.param = param
        self.train_mode = param['train_mode']
        self.log = param['log']
        self.device = param['device']
        self.batch_size = param['batch_size']
        self.lr = param['lr']
        self.rootpath = param['rootpath']
        self.data_name = param['data_name']
        self.propertys = param['propertys']
        self.window = param['window']
        self.shift = param['shift']
        self.ext = (2*param['ext']+1)**2
        self.ext1 = param['ext']
        self.t = param['t']
        self.pos = 0
        self.head_dim = {
            'ts_tcc_encoder':100,
            'simmtm_encoder':2688,
            'baseline_encoder':100*(self.window*2+1),
            'fe_encoder':(self.window*2+1)*128,
            'cnn1d_encoder':128*4,
            'transformer_encoder':64*1,
            'transformerT_encoder':64*1,
            'cnn2d_transformer_encoder':64*1,
            'transformerDouble_encoder':32*1,
            'Mid_transformer_encoder':64*1,
            'cnn_transformer_encoder':64*1,
            
        }
        
        self.ssl_method = param['ssl_method']
        self.ssl_encoder = param['ssl_encoder']
        self.ssl_paraname = param['ssl_paraname']
        self.ssl_epoch = param['ssl_epoch']
        self.ssl_encoder_ = encoder_dict(self.ssl_encoder,len(self.propertys),(self.ext),self.window*2+1+self.pos)
        
        self.ft_from_mode = param['ft_from_mode']
        self.ft_from_feature = param['ft_from_feature']
        self.ft_from_ssl_epoch = param['ft_from_ssl_epoch']
        self.ft_method = param['ft_method']
        self.ft_encoder = param['ft_encoder']
        self.ft_paraname = param['ft_paraname']
        self.ft_epoch = param['ft_epoch']
        self.ft_encoder_ = encoder_dict(self.ft_encoder,len(self.propertys),(self.ext),self.window*2+1+self.pos)
        
        self.pd_from_mode = param['pd_from_mode']
        self.pd_from_feature = param['pd_from_feature']
        self.pd_from_ft_epoch = param['pd_from_ft_epoch']
        self.pd_from_ssl_epoch = param['pd_from_ssl_epoch']
        self.pd_method = param['pd_method']
        self.pd_encoder = param['pd_encoder']
        self.pd_paraname = param['pd_paraname']
        self.pd_encoder_ = encoder_dict(self.pd_encoder,len(self.propertys),(self.ext),self.window*2+1+self.pos)
        
        if self.data_name == 'jintao':
                self.shape = (576,1767)
        elif self.data_name == 'boxing_es4cs2' :
            self.shape = (1121,881)
            self.n_class = 3
        elif self.data_name == 'boxing_es4cs4':
            self.shape = (1121,881)
            self.n_class = 4
        elif self.data_name == 'boxing_es4cs1':
            self.shape = (1121,881)
            self.n_class = 3
        elif self.data_name == 'shengli_ngs3':
            self.shape = (511,621)
            self.n_class = 3
        
        param_ts_tcc = {'head_dim':self.head_dim[self.ssl_encoder],
                         'encoder':self.ssl_encoder_,
                         'device':self.device,
                         'batchsize':self.batch_size,
                         'shape':self.shape,
                         'lr':self.lr}
        
        param_mask = {'head_dim':self.head_dim[self.ssl_encoder],
                        'input_channel':len(self.propertys)*(self.ext),
                         'encoder':self.ssl_encoder_,
                         'device':self.device,
                         'lr':self.lr,
                         'seq_len':2*self.window+1,
                         }
        param_simmtm2 = {'head_dim':self.head_dim[self.ssl_encoder],
                        'input_channel':1,
                         'encoder':encoder_dict(self.ssl_encoder,1,1,1),
                         'device':self.device,
                         'batchsize':self.batch_size,
                         'lr':self.lr,
                         'seq_len':2*self.window+1,
                         'positive_num':3}
        
        if self.train_mode == 'ssl':
            if self.ssl_method == 'ts_tcc':
                self.algorithm = algorithm_dict('ts_tcc',param_ts_tcc)
                self.algorithm.to(self.device)
            if self.ssl_method == 'ts_tcc_channelT':
                self.algorithm = algorithm_dict('ts_tcc_channelT',param_ts_tcc)
                self.algorithm.to(self.device)
            elif self.ssl_method == 'ts_tcc_ext':
                self.algorithm = algorithm_dict('ts_tcc_ext',param_ts_tcc)
                self.algorithm.to(self.device)
            elif self.ssl_method == 'ts_tcc_mask':
                self.algorithm = algorithm_dict('ts_tcc_mask',param_ts_tcc)
                self.algorithm.to(self.device)
            elif self.ssl_method == 'mask_channel_independent':
                self.algorithm = algorithm_dict('mask_channel_independent',param_mask)
                self.algorithm.to(self.device)
            elif self.ssl_method == 'simmtm_channel_independent':
                self.algorithm = algorithm_dict('simmtm_channel_independent',param_simmtm2)
                self.algorithm.to(self.device)
            elif self.ssl_method == 'pmsn':
                self.algorithm = algorithm_dict('pmsn',param_ts_tcc)
                self.algorithm.to(self.device)
            # logger
            self.savepath = os.path.join(self.rootpath,self.log, self.data_name, self.train_mode, 
                                        self.ssl_method, self.ssl_encoder, self.ssl_paraname)
        
        elif self.train_mode == 'ft':
            # Load Model
            if self.ft_from_mode == 'ssl':#load model from ssl
                self.savepath = os.path.join(self.rootpath,self.log,self.data_name,self.train_mode,
                                    self.ft_method,self.ft_encoder,self.ft_paraname,self.ft_from_mode,
                                    self.ssl_method,self.ssl_encoder,self.ssl_paraname, str(self.ft_from_ssl_epoch),
                                    self.ft_from_feature)
                self.model_path = os.path.join(self.rootpath,self.log,self.data_name,self.ft_from_mode,
                                    self.ssl_method,self.ssl_encoder,self.ssl_paraname,'model')
                chkpoint = torch.load(os.path.join(self.model_path, "checkpoint_{}.pt".format(self.ft_from_ssl_epoch)), map_location=self.device)
                print('Load Model:',os.path.join(self.model_path, "checkpoint_{}.pt".format(self.ft_from_ssl_epoch)))
                self.ft_encoder_[0].load_state_dict(chkpoint["fe"])
                self.ft_encoder_[1].load_state_dict(chkpoint["te"])
                
                # freezy layer,注意修改ft_paraname
                for encoder in self.ft_encoder_:
                    for param in encoder.parameters():
                        param.requires_grad = False
                #choose fe or fe+te feature in ssl to finetune
                if self.ft_from_feature == 'fe':
                    self.ft_encoder_c = nn.Sequential(self.ft_encoder_[0],origin()).to(self.device)
                    self.ssl_encoder = 'fe_encoder'
                else:
                    self.ft_encoder_c = nn.Sequential(self.ft_encoder_[0],self.ft_encoder_[1]).to(self.device)
            elif self.ft_from_mode == 'None':#train classifier directly
                print('Load Model: init model')
                self.savepath = os.path.join(self.rootpath,self.log,self.data_name,self.train_mode,
                                    self.ft_method,self.ft_encoder,self.ft_paraname,self.ft_from_mode,
                                    self.ft_from_feature)
                #use fe or fe+te feature to directly finetune
                if self.ft_from_feature == 'fe':
                    self.ft_encoder_c = nn.Sequential(self.ft_encoder_[0],origin()).to(self.device)
                    self.ssl_encoder = 'fe_encoder'
                else:
                    self.ft_encoder_c = nn.Sequential(self.ft_encoder_[0],self.ft_encoder_[1]).to(self.device)
            # Finetune Method
            if self.ft_method == 'simple_classifier':
                
                self.algorithm = simple_classifier(self.ft_encoder_c, self.n_class, self.head_dim[self.ssl_encoder],device=self.device,batch_size=128,lr=self.lr).to(self.device)
                      
        if self.train_mode == 'predict':
            # Load Model
            #use ssl model output to predict,we use kmeans or other unsupervised cluster method
            if self.pd_from_mode == 'ssl':
                
                self.model_path = os.path.join(self.rootpath,self.log,self.data_name,self.pd_from_mode,
                                           self.ssl_method,self.ssl_encoder,self.ssl_paraname,'model')
                self.savepath = os.path.join(self.rootpath,self.log,self.data_name,self.train_mode,
                                    self.pd_method,self.pd_encoder,self.pd_paraname,self.pd_from_mode,
                                    self.ssl_method, self.ssl_encoder,self.ssl_paraname, str(self.pd_from_ssl_epoch),
                                    self.pd_from_feature)
                chkpoint = torch.load(os.path.join(self.model_path, "checkpoint_{}.pt".format(self.pd_from_ssl_epoch)), map_location=self.device)
                print('Load Model:',os.path.join(self.model_path, "checkpoint_{}.pt".format(self.pd_from_ssl_epoch)))
                self.pd_encoder_[0].load_state_dict(chkpoint["fe"])
                self.pd_encoder_[1].load_state_dict(chkpoint["te"])
                self.cls = proj_head(self.head_dim[self.ssl_encoder],32,6,False)
                self.cls.load_state_dict(chkpoint["clf"])
                #choose fe or te+fe feature to cluster
                if self.pd_from_feature == 'fe':
                    self.pd_encoder_c = nn.Sequential(self.pd_encoder_[0],origin(),self.cls).to(self.device)
                else:
                    self.pd_encoder_c = nn.Sequential(self.pd_encoder_[0],self.pd_encoder_[1],self.cls).to(self.device)
            elif self.pd_from_mode == 'ft':
                #load model from ft, we need classifier in finetune to predict
                
                if self.ft_from_mode == 'None':#use directly finetune model without ssl to predict
                    self.savepath = os.path.join(
                                    self.rootpath,self.log,self.data_name,self.train_mode,
                                    self.pd_method,self.pd_encoder,self.pd_paraname,self.pd_from_mode,
                                    self.ft_method,self.ft_encoder,self.ft_paraname,self.ft_from_mode,
                                    self.ft_from_feature)
                    self.model_path = os.path.join(
                                    self.rootpath,self.log,self.data_name,self.pd_from_mode,
                                    self.ft_method,self.ft_encoder,self.ft_paraname,self.ft_from_mode,
                                    self.ft_from_feature,'model')
                else:#use finetune model from ssl to predict
                    self.model_path = os.path.join(
                                    self.rootpath,self.log,self.data_name,self.pd_from_mode,
                                    self.ft_method,self.ft_encoder,self.ft_paraname,self.ft_from_mode,
                                    self.ssl_method,self.ssl_encoder,self.ssl_paraname, str(self.pd_from_ssl_epoch),
                                    self.ft_from_feature,'model')
                                    
                    self.savepath = os.path.join(
                                    self.rootpath,self.log,self.data_name,self.train_mode,
                                    self.pd_method,self.pd_encoder,self.pd_paraname,self.pd_from_mode,
                                    self.ft_method,self.ft_encoder,self.ft_paraname,self.ft_from_mode,str(self.pd_from_ft_epoch),
                                    self.ssl_method,self.ssl_encoder,self.ssl_paraname, str(self.pd_from_ssl_epoch),
                                    self.ft_from_feature)
                chkpoint = torch.load(os.path.join(self.model_path, "checkpoint_{}.pt".format(self.pd_from_ft_epoch)), map_location=self.device)
                print('Load Model:',os.path.join(self.model_path, "checkpoint_{}.pt".format(self.pd_from_ft_epoch)))
                cls = proj_head(self.head_dim[self.ssl_encoder],64,self.n_class)
                #choose fe or fe+te feature in finetune model to predict,also use classifier in their model
                if self.ft_from_feature == 'fe':
                    self.ssl_encoder = 'fe_encoder'
                    cls = proj_head(self.head_dim[self.ssl_encoder],64,self.n_class)
                    self.pd_encoder_[0].load_state_dict(chkpoint["fe"])
                    cls.load_state_dict(chkpoint["clf"])
                    self.pd_encoder_c = nn.Sequential(self.pd_encoder_[0],origin(),cls).to(self.device)
                else:
                    print(self.head_dim[self.ssl_encoder],)
                    self.pd_encoder_[1].load_state_dict(chkpoint["te"])
                    self.pd_encoder_[0].load_state_dict(chkpoint["fe"])
                    cls.load_state_dict(chkpoint["clf"])
                    self.pd_encoder_c = nn.Sequential(self.pd_encoder_[0],self.pd_encoder_[1],cls).to(self.device)
            # use origin waveform to predict, usually use kmeans or other unsupervised method to predict
            elif self.pd_from_mode == 'None':
                print('Load Model: raw data')
                self.savepath = os.path.join(
                                self.rootpath,self.log,self.data_name,self.train_mode,
                                self.pd_method,self.pd_encoder,self.pd_paraname,self.pd_from_mode)
                self.pd_encoder_c = nn.Sequential(origin(),origin()).to(self.device)
            
            
                    
    def ssl_train(self):
        # animator = d2l.Animator(xlabel='epoch', xlim=[1, self.ft_epoch], ylim=[0.01, 1.1],
        #                         legend=['train loss'], figsize=(10, 6))
        #  init_model
        losses, self.ssl_model = self.algorithm.return_init()
        # Average meters
        loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
        # Load data
        self.train_dl,self.train_labe_dl = get_dataloader(self.data_name,shift=self.shift,window=self.window,propertys=self.propertys, train_mode=self.train_mode, 
                                         method=self.ssl_method, batch_size=self.batch_size,ext=self.ext1,t=self.t)
        
        copy_Files(self.savepath)
        print('***************************End CopyFiles**********************************************')
        os.makedirs(self.savepath+'/png/', exist_ok=True)
        self.logger, _ = starting_logs(self.data_name, self.ssl_method,self.ssl_encoder,self.ssl_paraname,
                                                           self.train_mode, self.savepath)
        self.logger.debug(self.param)
        # save
        self.model_dir = os.path.join(self.savepath,'model')
        os.makedirs(self.model_dir, exist_ok=True)
        save_checkpoint(self.model_dir, self.ssl_model, self.data_name, self.param, epoch = 0)
        # train
        loss = {}
        attn = []
        for epoch in range(1, self.ssl_epoch + 1):
            self.algorithm.train()
            for step, (data,(data_l,label)) in tqdm(enumerate(zip(self.train_dl,self.train_labe_dl))):
                data = to_device(data, self.device)
                data_l = to_device(data_l, self.device)
                label = to_device(label, self.device)
                losses, self.ssl_model = self.algorithm.update([data],step,epoch,self.savepath)
                for key, val in losses.items():
                    loss_avg_meters[key].update(val, self.batch_size)
                
            self.algorithm.update_lr(epoch)  
            attn.append(copy.deepcopy(self.algorithm.attn))# get attn
            dist = torch.cat(copy.deepcopy(self.algorithm.dist)).cpu().numpy()# get similarity distribute
            self.algorithm.reset_attn()
            
            if epoch % 5 == 0 and self.train_mode == 'ssl':
                save_checkpoint(self.model_dir, self.ssl_model, self.data_name, self.param, epoch )
                
            for key, val in loss_avg_meters.items():
                self.logger.debug(f'epoch:{epoch}/{self.ssl_epoch}, {key}\t: {val.avg:2.4f}')
                loss[key] = loss.get(key,[])+[val.avg]
                loss_avg_meters[key].reset()
            plt.cla()
            lossn = list(loss.keys())
            lossd = np.array(list(loss.values()))
            plt.figure(figsize=(5,5))
            plt.plot(lossd.T)
            plt.legend(lossn)
            plt.xlim([0,self.ssl_epoch])
            plt.ylim([0,10])
            plt.savefig(self.savepath+'/loss.png')
            np.save(self.savepath+'/loss.npy', lossd)
            
            
            plt.cla()
            attn_tensor = torch.stack(attn,dim=0)
            torch.save(attn_tensor,self.savepath+'/attn.pt')
            plt.figure(figsize=(13,5))
            plt.plot(attn_tensor)
            plt.xlim([0,self.ssl_epoch])
            plt.legend(self.propertys,bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
            plt.tight_layout()
            plt.savefig(self.savepath+'/attn.png')
            if epoch % 5 == 0 or epoch == 1:
                np.save(self.savepath+'/dist_{}.npy'.format(epoch),dist)
            
            self.logger.debug(f'-------------------------------------')
        
    def finetune(self):
        animator = d2l.Animator(xlabel='epoch', xlim=[1, self.ft_epoch], ylim=[0.01, 1.1],
                                legend=['train loss', 'test_auc', 'train_auc'], figsize=(10, 6))
        # Average meters
        loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
        # load data
        self.train_dl,self.test_dl = get_dataloader(self.data_name, shift=self.shift,window=self.window,propertys=self.propertys,train_mode=self.train_mode, batch_size=self.batch_size,method=self.ssl_method,ext=self.ext1,t=self.t)
        copy_Files(self.savepath)
        os.makedirs(self.savepath, exist_ok=True)
        self.logger, _ = starting_logs(self.data_name, self.ft_method,self.ft_encoder,self.ft_paraname,
                                                           self.train_mode, self.savepath)
        self.logger.debug(self.param)
        print("Save Path:", self.savepath)
        # init model
        losses, self.ft_model = self.algorithm.return_init()
        # save
        self.model_dir = os.path.join(self.savepath,'model')
        os.makedirs(self.model_dir, exist_ok=True)
        save_checkpoint(self.model_dir, self.ft_model, self.data_name, self.param, epoch = 0)
        # train
        self.best_f1 = 0
        self.best_acc = 0
        for epoch in range(1, self.ft_epoch + 1):
            self.algorithm.train()
            for step, data in tqdm(enumerate(self.train_dl)):
                data = to_device(data, self.device)
                losses, self.ft_model = self.algorithm.update(data)
                for key, val in losses.items():
                        loss_avg_meters[key].update(val, self.batch_size)  
            # evaluate            
            self.evaluate()
            self.calc_results_per_run()
            if epoch % 5 == 0:
                save_checkpoint(self.model_dir, self.ft_model, self.data_name, self.param, epoch = epoch)
            # if self.f1 > self.best_f1:  # save best model based on best f1.
            #     self.best_f1 = self.f1
            #     self.best_acc = self.acc
            #     save_checkpoint(self.model_dir, self.ft_model, self.data_name, self.param, epoch = epoch)
                
            loss_curve = tuple([i.avg for _, i in loss_avg_meters.items()])
            animator.add(epoch + 1, loss_curve)
            for key, val in loss_avg_meters.items():
                self.logger.debug(f'epoch:{epoch}/{self.ft_epoch}, {key}\t: {val.avg:2.4f}')
                self.logger.debug(f'Acc:{self.acc:2.4f} \t F1:{self.f1:2.4f} (best: {self.best_f1:2.4f})')
            self.logger.debug(f'-------------------------------------')
            plt.savefig(os.path.join(self.savepath,'loss.png'),dpi=200)
    
    def predict(self):
        self.pd_encoder_c.eval()
        self.train_dl,_ = get_dataloader(self.data_name, shift=self.shift,window=self.window,propertys=self.propertys,train_mode=self.train_mode, batch_size=self.batch_size,method=self.ssl_method,ext=self.ext1,t=self.t)
        copy_Files(self.savepath)
        os.makedirs(self.savepath, exist_ok=True)
        print("Save Path:", self.savepath)
        pos = None
        if 'boxing' in self.data_name or 'shengli' in self.data_name:
            _ , pos,_,_ = load_data(self.data_name,norm = True,window = 10, step=1,train_mode='ft',train_dataset='train',ep=0)
        if self.pd_method == 'kmeans':
            kmeans(self.pd_encoder_c,self.train_dl,self.device,self.shape,self.savepath,pos)
        elif self.pd_method == 'cls':
            cls(self.pd_encoder_c,self.cls,self.train_dl,self.device,self.shape,self.savepath,pos)
        elif self.pd_method == 'classifier':
            classifier(self.pd_encoder_c,self.train_dl,self.device,self.shape,self.savepath,pos)
            
    def evaluate(self):
        encoder = self.ft_model[0]
        token_transformer = self.ft_model[1]
        classifier = self.ft_model[2]
        encoder.eval()
        token_transformer.eval()
        classifier.eval()

        total_loss_ = []

        self.pred_labels = np.array([])
        self.true_labels = np.array([])

        with torch.no_grad():
            for data in self.test_dl:
                data_samples = to_device(data, self.device)
                data = data_samples['sample_ori'].float()
                labels = data_samples['label'].long()

                # forward pass
                x, _ = encoder(data)
                x, _ = token_transformer(x)
                x = x.reshape(x.shape[0],-1)
                predictions = classifier(x)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss_.append(loss.item())
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.pred_labels = np.append(self.pred_labels, pred.cpu().numpy())
                self.true_labels = np.append(self.true_labels, labels.data.cpu().numpy())

        self.trg_loss = torch.tensor(total_loss_).mean()  # average loss
        
    def calc_results_per_run(self):
        self.acc, self.f1 = calc_metrics(self.pred_labels, self.true_labels)

    
    # if step == 0:
    #     try:
    #         print(adasd)
    #         plt.cla()
    #         plt.plot(data["sample_ori"][0,0,:].reshape(-1,2*self.window+1).T.float().cpu())
    #         plt.savefig('ori_dat_{}.png'.format(step),dpi=200)
    #         plt.cla()
    #         plt.plot(data["transformed_samples"][0][0,0,:].reshape(-1,2*self.window+1).T.float().cpu())
    #         plt.savefig('strong_dat_{}.png'.format(step),dpi=200)
    #         plt.cla()
    #         plt.plot(data["transformed_samples"][1][0,0,:].reshape(-1,2*self.window+1).T.float().cpu())
    #         plt.savefig('weak_dat_{}.png'.format(step),dpi=200)
    #         plt.cla()
    #         ext = 9
    #         t = int(ext/3)
    #         mat = np.zeros(self.shape)
    #         for i in data["position"]:
    #             for i21 in range(-ext,ext+1,t):
    #                 for i22 in range(-ext,ext+1,t):
    #                     try:
    #                         mat[int(i[0].cpu().numpy())+i21,int(i[1].cpu().numpy())+i22] = 1
    #                     except:
    #                         pass
    #         plt.matshow(mat)
    #         plt.savefig('pos_{}.png'.format(step),dpi=200)
    #         plt.cla()
    #         # print('Draw samples successful!')
    #     # plt.scatter(data["position"][:,0],data["position"][:,1],s=10)
                
    #     except:
    #         pass