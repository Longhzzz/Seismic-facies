import numpy as np
import torch
import copy
from sklearn import preprocessing
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from augmentation import *
from tqdm import tqdm
import os
from jintao import *
from boxing import *
from shengli import *
def load_data(name,shift=0,norm = True,window = 10, step=1,propertys=['sx'],train_mode='ssl',train_dataset='train',ep=2,ext=1,t=1):
    if name == 'jintao':
        train_dat = []
        train_label_dat = []
        for property in propertys:
            dat ,train_pos, label_dat, label = jintao_data(name=name,window=window,step=step,norm=norm,shift=shift,property=property,train_mode=train_mode,ext=ext,t=t)
            train_dat.append(dat)
            train_label_dat.append(label_dat)
        train_dat = np.stack(train_dat,axis=1)
        train_label_dat = np.stack(train_label_dat,axis=1)
    if 'boxing' in name:
        train_dat = []
        train_pos = []
        for property in propertys:
            dat ,train_pos = boxing_data(name,window,step,norm,shift,ep,property,train_mode)
            train_dat.append(dat)
        train_dat = np.stack(train_dat,axis=1)
    if 'shengli' in name:
        train_dat = []
        train_label_dat = []
        for property in propertys:
            dat ,train_pos, label_dat, label = shengli_data(name=name,window=window,step=step,norm=norm,shift=shift,property=property,train_mode=train_mode,ext=ext,t=t)
            train_label_dat.append(label_dat)
            train_dat.append(dat)
        train_dat = np.stack(train_dat,axis=1)
        train_label_dat = np.stack(train_label_dat,axis=1)
        print('shengli:',train_dat.shape)
    return torch.from_numpy(train_dat), torch.from_numpy(train_pos),torch.from_numpy(train_label_dat),torch.from_numpy(label)
class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, train_mode, method, oversample=False):
        super(Load_Dataset, self).__init__()
        self.train_mode = train_mode
        self.method = method
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        if method in ['ts_tcc','mask_channel_independent','pmsn']:
            X_train = X_train.reshape(X_train.shape[0],-1,X_train.shape[-1])
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train).long()
        # if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        #     X_train = X_train.permute(0, 2, 1)
        if train_mode == 'ft':
            y_train = y_train[:,3]
        if train_mode == 'finetune' and oversample == True:
            self.x_data,self.y_data = get_balance_class_oversample(X_train, y_train)
            self.x_data = torch.from_numpy(self.x_data)
            self.y_data = torch.from_numpy(self.y_data)
        else:
            self.x_data = X_train
            self.y_data = y_train
        print("Train data shape:",self.x_data.shape)
        self.num_channels = X_train.shape[1]
        self.len = X_train.shape[0]
        self.window = self.gauss_window(X_train.shape[-1],2)
    def gauss_window(self,w,sigma):
        # Create a tensor for the window
        window = torch.arange(w).float()
        # Calculate the Gaussian function
        window = torch.exp(-0.5 * ((window - (w) // 2) / sigma) ** 2)
        # Normalize the window
        window /= window.sum()
        return window * w
    def __getitem__(self, index):
        inp = self.x_data[index]#attr*ext*seqlen
        shape = inp.shape
        # inp = inp.reshape(-1,shape[-1])#nvar * seqlen
        if self.train_mode == "ssl" :
            if self.method == 'ts_tcc' or self.method == 'ts_tcc_ext':
                # X0 = jitter(permutation(self.x_data[index], max_segments = 5), 0.8)
                # X1 = ts_tcc_scaling(self.x_data[index].unsqueeze(0), 0.3)
                X1 = jitter(inp, 0.2)
                X0 = jitter(DA_TimeWarp(inp, sigma=0.5),0.2)
                # X0 = jitter(generate_continuous_mask(inp, n=5, l=0.3),0.3)
                # X0 = self.x_data[index].squeeze(-1)
                # X1 = self.x_data[index].squeeze(-1)
                if len(X1.shape) == 3:
                    X1 = X1.reshape(-1,shape[-1])
                if len(X0.shape) == 3:
                    X0 = X0.reshape(-1,shape[-1])
                sample = {
                    'transformed_samples': [X0, X1] ,
                    'position': self.y_data[index],
                    'sample_ori':  inp}
            elif self.method == 'ts_tcc_channelT' or self.method == 'pmsn':
                # X1 = jitter(inp, 0.2)
                # X0 = DA_TimeWarp(inp, sigma=0.3)
                X1 = jitter(inp, 0.2) 
                X0 = jitter(inp, 0.2) 
                #X1 attr*ext*seqlen
                # X0 = data_transform_masked4cl(inp, 0.5, 3, positive_nums=1, distribution='geometric')
                # if len(X1.shape) == 3:
                #     X1 = X1.reshape(-1,shape[-1])
                # if len(X0.shape) == 3:
                #     X0 = X0.reshape(-1,shape[-1])
                sample = {
                    'transformed_samples': [X0, X1] ,
                    'position': self.y_data[index],
                    'sample_ori':  inp}
            elif self.method == 'ts_tcc_mask' :
                X1 = jitter(inp, 0.2)
                # X0 = DA_TimeWarp(inp, sigma=0.3)
                # X1 = jitter(inp, 0.2) 
                # X0 = jitter(inp, 0.2) 
                #X1 attr*ext*seqlen
                X0 = data_transform_masked4cl(inp, 0.5, 3, positive_nums=1, distribution='geometric')
                # if len(X1.shape) == 3:
                #     X1 = X1.reshape(-1,shape[-1])
                # if len(X0.shape) == 3:
                #     X0 = X0.reshape(-1,shape[-1])
                sample = {
                    'transformed_samples': [X0, X1] ,
                    'position': self.y_data[index],
                    'sample_ori':  inp}
            elif self.method == 'mask_channel_independent':
                X1 = generate_continuous_mask(inp, n=5, l=0.2)
                sample = {
                    'sample_mask': X1 ,
                    'position': self.y_data[index],
                    'sample_ori':  inp}
            else:
                sample = {
                    'sample_ori':  inp,
                    'position': self.y_data[index]}
        elif self.train_mode == "predict":
            sample = {
                'position': self.y_data[index] ,
                'sample_ori': inp}
        elif self.train_mode == 'ft':
            sample = {
                'label': self.y_data[index] ,
                'sample_ori': inp}
        return sample
    def __len__(self):
        return self.len

def get_dataloader(name, shift=-7,window=8,propertys=['sx'],train_mode='ssl', method='ts_tcc', batch_size=256,step=2,ep=2,ext=1,t=1):
    if train_mode == 'ssl':
        train_dat, train_pos,train_label_dat,train_label_label = load_data(name,shift,propertys=propertys,norm = True,train_mode=train_mode,window=window,step=step,ep=ep,ext=ext,t=t)
        train_dataset = {'samples':train_dat,'labels':train_pos}
        train_dataset = Load_Dataset(train_dataset,train_mode, method)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,collate_fn=custom_collate_fn,
                                                   shuffle=True, drop_last=True, num_workers=0)
        
        class_sample_count = np.array([len(np.where(train_label_label[:,2] == t)[0]) for t in np.unique(train_label_label[:,2])])
        print(class_sample_count)
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[int(t)] for t in train_label_label[:,2]])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        dataset_train = torch.utils.data.TensorDataset(train_label_dat, train_label_label)
        if len(train_label_dat)<59:
            train_label_dat = train_label_dat.repeat(6,1,1,1)
            train_label_label = train_label_label.repeat(6,1,1,1)
        test_loader = torch.utils.data.DataLoader(dataset_train, batch_size=60, sampler = None) 
        def cycle_dataloader(dataloader):
            while True:
                for batch in dataloader:
                    yield batch
        return train_loader, iter(cycle_dataloader(test_loader))
    elif train_mode == 'predict':
        train_dat, train_pos,_,_ = load_data(name,shift,propertys=propertys,norm = True,train_mode=train_mode,window=window,ep=ep,step=1,ext=ext,t=t)
        train_dataset = {'samples':train_dat,'labels':train_pos}
        train_dataset = Load_Dataset(train_dataset,train_mode, method)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                   shuffle=True, drop_last=False, num_workers=0)
        return train_loader, None
    elif train_mode == 'ft':
        train_dat, train_label = load_data(name,shift,propertys=propertys,norm = True,window = window,train_mode=train_mode,train_dataset='train',ep=ep,step=step,ext=ext,t=t)
        train_dataset = {'samples':train_dat,'labels':train_label}
        train_dataset = Load_Dataset(train_dataset,train_mode, method,oversample=False)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                   shuffle=True, drop_last=True, num_workers=0)
        
        test_dat, test_label = load_data(name,shift,propertys=propertys,norm = True,window = window, train_mode=train_mode,train_dataset='test',ep=ep,step=step,ext=ext,t=t)
        test_dataset = {'samples':test_dat,'labels':test_label}
        test_dataset = Load_Dataset(test_dataset,train_mode, method)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                   shuffle=True, drop_last=False, num_workers=0)
        return train_loader, test_loader
def custom_collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        dict = {}
        for key in elem:
            if 'transform' in key:
                d = custom_collate_fn([d[key] for d in batch])
                
                # n0 = torch.randint(1,d[0].shape[1]+1,size=(1,)).item()
                # X0 = d[0][:,torch.randperm(d[0].shape[1])[:n0].tolist()]
                
                # n1 = torch.randint(1,d[0].shape[1]+1,size=(1,)).item()
                # # n1 = d[0].shape[1]+1
                # X1 = d[1][:,torch.randperm(d[0].shape[1])[:n1].tolist()]
                
                # dict[key] = [X0,X1]
                
                dict[key] = d
            else:
                dict[key] = custom_collate_fn([d[key] for d in batch])
        return dict
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [custom_collate_fn(samples) for samples in transposed]
    
    
    
    
    
    
    
    
