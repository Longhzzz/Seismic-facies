import copy
import random
import torch
import torch.nn as nn
import numpy as np
import os
import warnings
from model import *
from utils import *
warnings.simplefilter(action='ignore', category=FutureWarning)
from einops import rearrange
from loss import *
from layers.SelfAttention_Family import *
from layers.Transformer_EncDec import *
from layers.Embed import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from augmentation import data_transform_masked4cl
from loss import ContrastiveWeight, AggregationRebuild, AutomaticWeightedLoss
import seaborn as sns 
from sklearn.manifold import TSNE
from kmeans_pytorch import kmeans, kmeans_predict

class ts_tcc(torch.nn.Module):
    def __init__(self, head_dim, encoder, device='cuda:0',batch_size=256, lr = 1e-5):
        super(ts_tcc, self).__init__()
        self.encoder = encoder[0]
        self.token_transformer = encoder[1]
        self.projection_head = proj_head(head_dim,128,64)
        self.network = nn.Sequential(self.encoder, self.token_transformer, self.projection_head)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=3e-4,
            betas=(0.9, 0.99)
        )
        
        # decayRate = 0.96
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)

        self.contrastive_loss = NTXentLoss(device, batch_size, 0.2, True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
    def return_init(self):
        return {'Total_loss': 0}, \
               [self.encoder, self.token_transformer, self.projection_head]

    def update(self, samples):
        aug1 = samples["transformed_samples"][0].float()
        aug2 = samples["transformed_samples"][1].float()
        self.optimizer.zero_grad()

        x1,_ = self.encoder(aug1)
        x2,_ = self.encoder(aug2)
        ct1,_ = self.token_transformer(x1)
        ct2,_ = self.token_transformer(x2)
    
        ct1 = self.flatten(ct1)
        ct2 = self.flatten(ct2)
        z1 = self.projection_head(ct1)
        z2 = self.projection_head(ct2)
        
        loss = self.contrastive_loss(z1, z2) 
        loss.backward()
        self.optimizer.step()
        return {'Total_loss': loss.item()}, \
               [self.encoder, self.token_transformer, self.projection_head]
    def output(self,samples):
        pass
    
class pmsn(torch.nn.Module):
    def __init__(self, head_dim, encoder, shape, device='cuda:0',batch_size=256, lr = 1e-5):
        super(pmsn, self).__init__()
        self.encoder = encoder[0]
        self.token_transformer = encoder[1]
        self.projection_head = proj_head(head_dim,128,64)
        
        self.prototypes = nn.Linear(64, 6, bias=False).weight
         
        self.anchor_encoder = copy.deepcopy(self.encoder)
        self.anchor_token_transformer = copy.deepcopy(self.token_transformer)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)
        self.network = nn.Sequential(self.anchor_encoder, self.anchor_token_transformer, 
                                     self.anchor_projection_head)

        self.optimizer = torch.optim.Adam(
            [*list(self.anchor_encoder.parameters()),
             *list(self.anchor_token_transformer.parameters()),
             *list(self.anchor_projection_head.parameters()),
             self.prototypes],
            lr=lr,
            weight_decay=3e-4,
            betas=(0.9, 0.99)
        )
        # decayRate = 0.96
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)

        self.cluster_loss = MSNLoss(temperature=0.2,c='None',power_law_exponent=2) 
        self.labeled_conloss = W_NTXentLoss_l(device, 0.2, True,shape=shape)
        # self.contrastive_loss = NTXentLoss(device, batch_size, 0.2, True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
    def return_init(self):
        return {'Total_loss': 0}, \
               [self.encoder, self.token_transformer, self.projection_head]
    

    def update(self, samples, step, epoch,path):
        aug1 = samples[0]["transformed_samples"][0].float()
        aug2 = samples[0]["transformed_samples"][1].float()
        #anchor : need to backforward,no anchor : need to momentum, weak aug : put in no anchor
        self.optimizer.zero_grad()
        self.update_momentum(self.anchor_encoder,self.encoder,0.996)
        self.update_momentum(self.anchor_token_transformer,self.token_transformer,0.996)
        self.update_momentum(self.anchor_projection_head,self.projection_head,0.996)
        
        x1,_ = self.anchor_encoder(aug1)
        x2,_ = self.encoder(aug2)
        x1 = x1.transpose(1,2)#b * len * nvar
        x2 = x2.transpose(1,2)#b * len * nvar
        ct1,_ = self.anchor_token_transformer(x1)
        ct2,_ = self.token_transformer(x2)
    
        ct1 = self.flatten(ct1)
        ct2 = self.flatten(ct2)
        z1 = self.anchor_projection_head(ct1)
        z2 = self.projection_head(ct2)
        loss_c = self.cluster_loss(z1,z2,self.prototypes,step, epoch,path) 
        loss_l = torch.tensor(0)
        if len(samples)>1:
            data_l = samples[1].float()
            data_l = data_l.reshape(data_l.shape[0],-1,data_l.shape[-1])
            label = samples[2]
            x3,_ = self.anchor_encoder(data_l)
            x3 = x3.transpose(1,2)
            ct3,_ = self.anchor_token_transformer(x3)
        
            ct3 = self.flatten(ct3)
            z3 = self.anchor_projection_head(ct3)
            loss_l = self.labeled_conloss(z3, label[:,2], label[:,0:2]) 
        loss = loss_c + loss_l
        loss.backward()
        self.optimizer.step()
        return {'Total_loss': loss.item()}, \
               [self.encoder, self.token_transformer, self.projection_head]
               
    @torch.no_grad()
    def update_momentum(self,model, model_ema, m):
        for model_ema, model in zip(model_ema.parameters(), model.parameters()):
            model_ema.data = model_ema.data * m + model.data * (1.0 - m)
               
class ts_tcc_channelT(torch.nn.Module):
    def __init__(self, head_dim, encoder, shape, device='cuda:0',batch_size=256, lr = 1e-5):
        super(ts_tcc_channelT, self).__init__()
        self.encoder = encoder[0]
        self.token_transformer = encoder[1]
        self.device = device
        self.projection_head = proj_head(head_dim,32,16)
        # self.projection_head = nn.Linear(head_dim,16)
        # self.projection_head = nn.Identity()
        self.cls = proj_head(head_dim,32,6,False)
        self.network = nn.Sequential(self.encoder, self.token_transformer, self.projection_head,self.cls)
        # self.optimizer = torch.optim.Adam(
        #     self.network.parameters(),
        #     lr=lr,
        #     weight_decay=3e-4,
        #     betas=(0.9, 0.99)
        # )
        self.optimizer = LARS(
            [params for params in self.network.parameters()],
            lr=0.2,
            weight_decay=1e-6,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )
        # "decay the learning rate with the cosine decay schedule without restarts"
        self.warmupscheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)
        self.mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)
        # decayRate = 0.96
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)

        self.contrastive_loss = W_NTXentLoss(device, batch_size, 0.2, True,shape=shape)
        self.labeled_conloss = W_NTXentLoss_l(device, 0.2, True,shape=shape)
        self.cluster_loss = NTXentLoss(device,6,0.2,True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.shape = shape
        self.flatten = nn.Flatten()
        self.sim = torch.nn.CosineSimilarity(dim=-1)
        self.sim2 = torch.nn.PairwiseDistance(p=2)
        self.dcl = DCLW(device, batch_size, 0.2, True,shape=shape)
        self.attn = None
        self.dist = None
        
    def return_init(self):
        return {'Total_loss': 0}, \
               [self.encoder, self.token_transformer, self.cls]
    def update(self, samples, step, epoch,path):
        aug1 = samples[0]["transformed_samples"][0].float()#b * nvar * ext* len
        aug2 = samples[0]["transformed_samples"][1].float()
        bs,nvar,ext,seqlen = aug1.shape
        position = samples[0]['position']#b*5
        self.optimizer.zero_grad()
        x1,_ = self.encoder(aug1,mode='notrain')
        x2,_ = self.encoder(aug2,mode='notrain')
        bs,nvar2,ext,seqlen = x1.shape
        
        # ############################no transformer#######################################
        # rand1 = torch.rand((bs,nvar))
        # mask1 = (rand1 <= 0.5)
        # mask1 = mask1.unsqueeze(-1).repeat(1,1,nvar2//nvar).reshape(bs,-1).unsqueeze(-1).repeat(1,1,17).unsqueeze(2).float().to(x1.device)
        # x1 = x1 * mask1
        # x11,_ = torch.max(x1,dim=1)
        # x11 = self.flatten(x11)
        
        # rand2 = torch.rand((bs,nvar))
        # mask2 = (rand2 <= 0.5)
        # mask2 = mask2.unsqueeze(-1).repeat(1,1,nvar2//nvar).reshape(bs,-1).unsqueeze(-1).repeat(1,1,17).unsqueeze(2).float().to(x1.device)
        # x2 = x2 * mask2
        # x22,_ = torch.max(x2,dim=1)
        # x22 = self.flatten(x22)
        # x11 = self.projection_head(x11)
        # x22 = self.projection_head(x22)
        # #loss_c = self.dcl(x11, x22, position,step,epoch,path) and self.projection_head = proj_head(17,32,16) and
        # #z1 = self.projection_head(ct1) z2 = self.projection_head(ct2)
        # ###################################################################################
        
        rand1 = torch.rand((bs,nvar))
        mask1 = (rand1 <= 0.5)
        mask1 = mask1.unsqueeze(-1).repeat(1,1,nvar2//nvar).reshape(bs,-1)
        mask1 = torch.cat([torch.zeros((bs,1)).bool(),mask1],dim=1)
        mask1 = mask1.unsqueeze(1).to(aug1.device).repeat(1,nvar2+1,1).unsqueeze(1)
        
        rand2 = torch.rand((bs,nvar))
        mask2 = (rand2 <= 0.5)
        mask2 = mask2.unsqueeze(-1).repeat(1,1,nvar2//nvar).reshape(bs,-1)
        mask2 = torch.cat([torch.zeros((bs,1)).bool(),mask2],dim=1)
        mask2 = mask2.unsqueeze(1).to(aug2.device).repeat(1,nvar2+1,1).unsqueeze(1)
        
        ct1,attn1 = self.token_transformer(x1,epoch=epoch,attn_mask = mask1)
        ct2,attn2 = self.token_transformer(x2,epoch=epoch,attn_mask = mask2)
        # attn1 = attn1[-1].mean(1).mean(0)[0,1:]
        attn1 = torch.cat(attn1,1).mean(1).mean(0)[0,1:]
        attn1= attn1.reshape(nvar,nvar2//nvar).sum(1)

        if self.attn is None:
            self.attn = attn1.cpu().detach()
        else:
            self.attn = self.attn + attn1.cpu().detach()
            
        
        ct1 = self.flatten(ct1)
        ct2 = self.flatten(ct2)
        
        ct1_copy = ct1.clone().detach()
        dist = self.sim(ct1_copy.unsqueeze(1), ct1_copy.unsqueeze(0))[~np.eye(bs, dtype=bool)]
        if self.dist is None:
            self.dist = [dist[:100000]]
        else:
            self.dist = self.dist + [dist[:100000]]
            
        if step == 0 :
            cluster_ids_x, cluster_centers = kmeans(
                X=ct2, num_clusters=6, distance='euclidean', device=self.device
            )
            # cluster_ids_y = kmeans_predict(
            #     y, cluster_centers, 'euclidean', device=self.device
            # )
            # print(self.sim(cluster_centers.unsqueeze(1),cluster_centers.unsqueeze(0)))
            
        z1 = self.projection_head(ct1)
        z2 = self.projection_head(ct2)
        
        # c1 = self.cls(ct1)
        # c2 = self.cls(ct2)
        
        # ct1 = F.normalize(ct1, dim=1)
        # ct2 = F.normalize(ct2, dim=1)
        if step == 0 and (epoch%5==0 or epoch==1):
            # z1_copy = z1.clone().detach().cpu()
            # ct2_copy = ct2.clone().detach().cpu()
            # z2_copy = z2.clone().detach().cpu()
            # postion_copy = position.clone().detach().cpu()
            # diff = postion_copy.unsqueeze(1) - postion_copy.unsqueeze(0)
            # distance_matrix = torch.sqrt(torch.sum(diff**2, dim=-1))/torch.sqrt(torch.tensor(self.shape[0])**2+torch.tensor(self.shape[1])**2)
            # plt.cla()
            # plt.figure(figsize=(5,5))
            # sns.distplot(distance_matrix)
            # plt.xlim([0,1])
            # plt.savefig(path+'/png/distance_{}.png'.format(epoch))
            plt.cla()
            plt.figure(figsize=(5,5))
            sns.distplot(dist.cpu())
            plt.xlim([-1,1])
            plt.savefig(path+'/png/dist_{}.png'.format(epoch))
            # plt.cla()
            # plt.figure(figsize=(5,5))
            # sns.distplot(self.sim(z1_copy.unsqueeze(1), z1_copy.unsqueeze(0))[0:512,:])
            # plt.xlim([-1,1])
            # plt.savefig(path+'/png/dist_proj_{}.png'.format(epoch))
            # plt.cla()
            # plt.figure(figsize=(5,5))
            # sns.distplot(self.sim(ct1_copy, ct2_copy))
            # plt.xlim([-1,1])
            # plt.savefig(path+'/png/dist_positive_{}.png'.format(epoch))
            # plt.cla()
            # plt.figure(figsize=(5,5))
            # sns.distplot(self.sim(z1_copy, z2_copy))
            # plt.xlim([-1,1])
            # plt.savefig(path+'/png/dist_proj_positive_{}.png'.format(epoch))
            # plt.cla()
            # plt.figure(figsize=(5,5))
            # ct1_std = ct1_copy.std(dim=0)
            # z1_std = z1_copy.std(dim=0)
            # sns.distplot(ct1_std)
            # sns.distplot(z1_std)
            # plt.legend(['ct','z'])
            # plt.xlim([0,1])
            # plt.savefig(path+'/png/dist_std_{}.png'.format(epoch))
            # plt.cla()
            
            # ct1_emb = TSNE(n_components=2,verbose=False).fit_transform(ct1_copy)
            # plt.scatter(ct1_emb[:,0],ct1_emb[:,1],s=5)
            # plt.savefig(path+'/png/fea_tsne{}.png'.format(epoch))
            # plt.cla()
            # z1_emb = TSNE(n_components=2,verbose=False).fit_transform(z1_copy)
            # plt.scatter(z1_emb[:,0],z1_emb[:,1],s=5)
            # plt.savefig(path+'/png/fea_proj_tsne{}.png'.format(epoch))
            # plt.cla()
            
        loss_l = torch.tensor(0)
        loss_cluster = torch.tensor(0)
        loss_smooth = torch.tensor(0)
        # cta1 = cta1.reshape(b,int(math.sqrt(ext)),int(math.sqrt(ext)),-1)
        # cta2 = cta2.reshape(b,int(math.sqrt(ext)),int(math.sqrt(ext)),-1)
        # d1 = torch.gradient(cta1,dim=1)[0]
        # d2 = torch.gradient(cta1,dim=2)[0]
        # loss_smooth = (torch.square(d1).sum(-1).mean() + torch.square(d2).sum(-1).mean())/2
            
        loss_c = self.dcl(z1, z2, position,step,epoch,path) 
        # prob = torch.softmax(c2/0.2,dim=1)
        # print()
        # # # print(torch.max(prob,dim=1))
        # ma,ind = torch.max(prob,dim=1)
        # print(prob.mean(0))
        # print(torch.bincount(ind))
        # # print(ind)
        # entropy = - torch.sum(prob * torch.log(prob+1e-5), dim=1).mean()
        # entropy2 = -  torch.sum(prob.mean(0) * torch.log(prob.mean(0) + 1e-5))
        # loss_cluster = (self.cluster_loss(c1.transpose(0,1),c2.transpose(0,1),step,epoch,path) - entropy2)
        if len(samples)>1:
            data_l = samples[1].float()
            data_l = data_l.reshape(data_l.shape[0],-1,data_l.shape[-1])
            label = samples[2]
            x3,_ = self.encoder(data_l)
            x3 = x3.transpose(1,2)
            ct3,_ = self.token_transformer(x3)
        
            ct3 = self.flatten(ct3)
            z3 = self.projection_head(ct3)
            loss_l = self.labeled_conloss(z3, label[:,2], label[:,-2:W]) 
        loss = loss_c + loss_l + loss_cluster + loss_smooth
        loss.backward()
        self.optimizer.step()
        return {'Total_loss': loss.item(),'cons_loss':loss_c.item(),'label_loss':loss_l.item(),'cluster_loss':loss_cluster.item(),'smooth_loss':loss_smooth.item()}, \
               [self.encoder, self.token_transformer, self.cls]
    def labeled_update(self,data,label):
        data = data.float().reshape(data.shape[0],-1,data.shape[-1])
        self.optimizer.zero_grad()
        x1,_ = self.encoder(data)
        x1 = x1.transpose(1,2)
        ct1,_ = self.token_transformer(x1)
    
        ct1 = self.flatten(ct1)
        z1 = self.projection_head(ct1)
        loss = self.labeled_conloss(z1, label[:,2], label[:,0:2]) 
        loss.backward()
        self.optimizer.step()
        return {'Total_label_loss': loss.item()}, \
               [self.encoder, self.token_transformer, self.projection_head]
    def reset_attn(self):
        self.attn = None
        self.dist = None
    def update_lr(self,epoch):
        if epoch < 10:
            self.warmupscheduler.step()
        if epoch >= 10:
            self.mainscheduler.step()
        
class ts_tcc_mask(torch.nn.Module):
    def __init__(self, head_dim, encoder, shape, device='cuda:0',batch_size=256, lr = 1e-5):
        super(ts_tcc_mask, self).__init__()
        self.encoder = encoder[0]
        self.token_transformer = encoder[1]
        self.device = device
        self.projection_head = proj_head(head_dim,32,16)
        # self.projection_head = nn.Linear(head_dim,16)
        # self.projection_head = nn.Identity()
        self.cls = proj_head(head_dim,32,6,False)
        self.network = nn.Sequential(self.encoder, self.token_transformer, self.projection_head,self.cls)
        # self.optimizer = torch.optim.Adam(
        #     self.network.parameters(),
        #     lr=lr,
        #     weight_decay=3e-4,
        #     betas=(0.9, 0.99)
        # )
        self.optimizer = LARS(
            [params for params in self.network.parameters()],
            lr=0.2,
            weight_decay=1e-6,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )
        # "decay the learning rate with the cosine decay schedule without restarts"
        self.warmupscheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)
        self.mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)
        # decayRate = 0.96
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)

        self.contrastive_loss = W_NTXentLoss(device, batch_size, 0.2, True,shape=shape)
        self.labeled_conloss = W_NTXentLoss_l(device, 0.2, True,shape=shape)
        self.cluster_loss = NTXentLoss(device,6,0.2,True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.shape = shape
        self.flatten = nn.Flatten()
        self.sim = torch.nn.CosineSimilarity(dim=-1)
        self.sim2 = torch.nn.PairwiseDistance(p=2)
        self.dcl = DCLW(device, batch_size, 0.2, True,shape=shape)
        self.attn = None
        self.dist = None
        
    def return_init(self):
        return {'Total_loss': 0}, \
               [self.encoder, self.token_transformer, self.cls]
    def update(self, samples, step, epoch,path):
        aug1 = samples[0]["transformed_samples"][0].float()#b * nvar * ext* len
        aug2 = samples[0]["transformed_samples"][1].float()
        bs,nvar,ext,seqlen = aug1.shape
        position = samples[0]['position']#b*5
        self.optimizer.zero_grad()
        x1,_ = self.encoder(aug1,mode='notrain')
        x2,_ = self.encoder(aug2,mode='notrain')
        bs,nvar2,ext,seqlen = x1.shape
        
        ct1,attn1 = self.token_transformer(x1,epoch=epoch,attn_mask = None)
        ct2,attn2 = self.token_transformer(x2,epoch=epoch,attn_mask = None)
        attn1 = torch.cat(attn1,1).mean(1).mean(0)[0,1:]
        attn1= attn1.reshape(nvar,nvar2//nvar).mean(1)
        if self.attn is None:
            self.attn = attn1.cpu().detach()
        else:
            self.attn = self.attn + attn1.cpu().detach()
            
        
        ct1 = self.flatten(ct1)
        ct2 = self.flatten(ct2)
        
        ct1_copy = ct1.clone().detach()
        dist = self.sim(ct1_copy.unsqueeze(1), ct1_copy.unsqueeze(0))[~np.eye(bs, dtype=bool)]
        if self.dist is None:
            self.dist = [dist[:100000]]
        else:
            self.dist = self.dist + [dist[:100000]]
            
        if step == 0 :
            cluster_ids_x, cluster_centers = kmeans(
                X=ct2, num_clusters=6, distance='euclidean', device=self.device
            )
            # cluster_ids_y = kmeans_predict(
            #     y, cluster_centers, 'euclidean', device=self.device
            # )
            # print(self.sim(cluster_centers.unsqueeze(1),cluster_centers.unsqueeze(0)))
            
        z1 = self.projection_head(ct1)
        z2 = self.projection_head(ct2)
        
        # c1 = self.cls(ct1)
        # c2 = self.cls(ct2)
        
        # ct1 = F.normalize(ct1, dim=1)
        # ct2 = F.normalize(ct2, dim=1)
        if step == 0 and (epoch%5==0 or epoch==1):
            # z1_copy = z1.clone().detach().cpu()
            # ct2_copy = ct2.clone().detach().cpu()
            # z2_copy = z2.clone().detach().cpu()
            # postion_copy = position.clone().detach().cpu()
            # diff = postion_copy.unsqueeze(1) - postion_copy.unsqueeze(0)
            # distance_matrix = torch.sqrt(torch.sum(diff**2, dim=-1))/torch.sqrt(torch.tensor(self.shape[0])**2+torch.tensor(self.shape[1])**2)
            # plt.cla()
            # plt.figure(figsize=(5,5))
            # sns.distplot(distance_matrix)
            # plt.xlim([0,1])
            # plt.savefig(path+'/png/distance_{}.png'.format(epoch))
            plt.cla()
            plt.figure(figsize=(5,5))
            sns.distplot(dist.cpu())
            plt.xlim([-1,1])
            plt.savefig(path+'/png/dist_{}.png'.format(epoch))
            # plt.cla()
            # plt.figure(figsize=(5,5))
            # sns.distplot(self.sim(z1_copy.unsqueeze(1), z1_copy.unsqueeze(0))[0:512,:])
            # plt.xlim([-1,1])
            # plt.savefig(path+'/png/dist_proj_{}.png'.format(epoch))
            # plt.cla()
            # plt.figure(figsize=(5,5))
            # sns.distplot(self.sim(ct1_copy, ct2_copy))
            # plt.xlim([-1,1])
            # plt.savefig(path+'/png/dist_positive_{}.png'.format(epoch))
            # plt.cla()
            # plt.figure(figsize=(5,5))
            # sns.distplot(self.sim(z1_copy, z2_copy))
            # plt.xlim([-1,1])
            # plt.savefig(path+'/png/dist_proj_positive_{}.png'.format(epoch))
            # plt.cla()
            # plt.figure(figsize=(5,5))
            # ct1_std = ct1_copy.std(dim=0)
            # z1_std = z1_copy.std(dim=0)
            # sns.distplot(ct1_std)
            # sns.distplot(z1_std)
            # plt.legend(['ct','z'])
            # plt.xlim([0,1])
            # plt.savefig(path+'/png/dist_std_{}.png'.format(epoch))
            # plt.cla()
            
            # ct1_emb = TSNE(n_components=2,verbose=False).fit_transform(ct1_copy)
            # plt.scatter(ct1_emb[:,0],ct1_emb[:,1],s=5)
            # plt.savefig(path+'/png/fea_tsne{}.png'.format(epoch))
            # plt.cla()
            # z1_emb = TSNE(n_components=2,verbose=False).fit_transform(z1_copy)
            # plt.scatter(z1_emb[:,0],z1_emb[:,1],s=5)
            # plt.savefig(path+'/png/fea_proj_tsne{}.png'.format(epoch))
            # plt.cla()
            
        loss_l = torch.tensor(0)
        loss_cluster = torch.tensor(0)
        loss_smooth = torch.tensor(0)
        # cta1 = cta1.reshape(b,int(math.sqrt(ext)),int(math.sqrt(ext)),-1)
        # cta2 = cta2.reshape(b,int(math.sqrt(ext)),int(math.sqrt(ext)),-1)
        # d1 = torch.gradient(cta1,dim=1)[0]
        # d2 = torch.gradient(cta1,dim=2)[0]
        # loss_smooth = (torch.square(d1).sum(-1).mean() + torch.square(d2).sum(-1).mean())/2
            
        loss_c = self.dcl(z1, z2, position,step,epoch,path) 
        # prob = torch.softmax(c2/0.2,dim=1)
        # print()
        # # # print(torch.max(prob,dim=1))
        # ma,ind = torch.max(prob,dim=1)
        # print(prob.mean(0))
        # print(torch.bincount(ind))
        # # print(ind)
        # entropy = - torch.sum(prob * torch.log(prob+1e-5), dim=1).mean()
        # entropy2 = -  torch.sum(prob.mean(0) * torch.log(prob.mean(0) + 1e-5))
        # loss_cluster = (self.cluster_loss(c1.transpose(0,1),c2.transpose(0,1),step,epoch,path) - entropy2)
        if len(samples)>1:
            data_l = samples[1].float()
            data_l = data_l.reshape(data_l.shape[0],-1,data_l.shape[-1])
            label = samples[2]
            x3,_ = self.encoder(data_l)
            x3 = x3.transpose(1,2)
            ct3,_ = self.token_transformer(x3)
        
            ct3 = self.flatten(ct3)
            z3 = self.projection_head(ct3)
            loss_l = self.labeled_conloss(z3, label[:,2], label[:,-2:W]) 
        loss = loss_c + loss_l + loss_cluster + loss_smooth
        loss.backward()
        self.optimizer.step()
        return {'Total_loss': loss.item(),'cons_loss':loss_c.item(),'label_loss':loss_l.item(),'cluster_loss':loss_cluster.item(),'smooth_loss':loss_smooth.item()}, \
               [self.encoder, self.token_transformer, self.cls]
    def labeled_update(self,data,label):
        data = data.float().reshape(data.shape[0],-1,data.shape[-1])
        self.optimizer.zero_grad()
        x1,_ = self.encoder(data)
        x1 = x1.transpose(1,2)
        ct1,_ = self.token_transformer(x1)
    
        ct1 = self.flatten(ct1)
        z1 = self.projection_head(ct1)
        loss = self.labeled_conloss(z1, label[:,2], label[:,0:2]) 
        loss.backward()
        self.optimizer.step()
        return {'Total_label_loss': loss.item()}, \
               [self.encoder, self.token_transformer, self.projection_head]
    def reset_attn(self):
        self.attn = None
        self.dist = None
    def update_lr(self,epoch):
        if epoch < 10:
            self.warmupscheduler.step()
        if epoch >= 10:
            self.mainscheduler.step()
            
class ts_tcc_ext(torch.nn.Module):
    def __init__(self, head_dim, encoder, device='cuda:0',batch_size=256, lr = 1e-5):
        super(ts_tcc_ext, self).__init__()
        self.encoder = encoder[0]
        self.token_transformer = encoder[1]
        
        self.projection_head = proj_head(head_dim,128,64)
        self.mask = get_mask(49).to(device)
        self.network = nn.Sequential(self.encoder, self.token_transformer, self.projection_head)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=3e-4,
            betas=(0.9, 0.99)
        )
        # decayRate = 0.96
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)

        self.contrastive_loss = NTXentLoss(device, batch_size, 0.2, True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.sim = torch.nn.CosineSimilarity(dim=-1)
    def return_init(self):
        return {'Total_loss': 0}, \
               [self.encoder, self.token_transformer, self.projection_head]

    def update(self, samples):
        aug1 = samples["transformed_samples"][0].float()
        aug2 = samples["transformed_samples"][1].float()
        self.optimizer.zero_grad()
        b, nvar, ext, seqlen = aug1.shape 
        x1 = aug1.permute(0, 2, 1, 3) # b, ext, nvar, seqlen
        x2 = aug2.permute(0, 2, 1, 3) # b, ext, nvar, seqlen
        x1 = x1.reshape(-1, nvar, seqlen)#b*ext, nvar, seqlen
        x2 = x2.reshape(-1, nvar, seqlen)#b*ext, nvar, seqlen
        
        x1,_ = self.encoder(x1)# b*ext, channel, len
        x2,_ = self.encoder(x2)# b*ext, channel, len
        ct1,_ = self.token_transformer(x1)# b*ext, channel, len
        ct2,_ = self.token_transformer(x2)# b*ext, channel, len
    
        ct1 = self.flatten(ct1)# b*ext, channel* len
        ct2 = self.flatten(ct2)# b*ext, channel* len
        z1 = self.projection_head(ct1)# b*ext, dim
        z2 = self.projection_head(ct2)# b*ext, dim
        
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        a1 = z1.reshape(b,ext,-1)# b,ext, dim
        a2 = z2.reshape(b,ext,-1)# b,ext, dim
        
        c1 = a1[:,int((ext+1)/2),:]
        c2 = a2[:,int((ext+1)/2),:]
        
        s1 = self.sim(a2.unsqueeze(2),a2.unsqueeze(1))
        if torch.rand(1)[0]<0.1:
            plt.imshow(s1[0,24].reshape(7,7).cpu().detach().numpy())
            plt.savefig('similiriaty.png')
            plt.cla()
        s_m = s1*self.mask
        loss_m = ext-torch.norm(s_m,p=float('inf'),dim=-1).sum()/(ext*b)
        
        loss_c = self.contrastive_loss(c1, c2)
        a = 0.5
        loss = a*loss_c + (1-a)*loss_m
        
        loss.backward()
        self.optimizer.step()
        return {'Total_loss': loss.item(),'Contrastive Loss':loss_c.item(),'Norm Loss':loss_m.item()}, \
               [self.encoder, self.token_transformer, self.projection_head]
    def output(self,samples):
        pass   
               
class mask_channel_independent(torch.nn.Module):
    def __init__(self, head_dim, input_channel,encoder, seq_len,device='cuda:0', lr = 1e-5):
        super(mask_channel_independent, self).__init__()
        self.encoder = encoder[0]
        self.token_transformer = encoder[1]
        self.device = device
        self.projection_head = proj_head(head_dim,256,128)
        self.network = nn.Sequential(self.encoder, self.token_transformer, self.projection_head)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=3e-4,
            betas=(0.9, 0.99)
        )
        self.flatten = nn.Flatten()
        self.decoder = proj_head(head_dim,1024,seq_len*input_channel)
        self.mse = torch.nn.MSELoss()
    def return_init(self):
        return {'Total_loss': 0}, \
               [self.encoder, self.token_transformer, self.projection_head]

    def update(self, samples):
        data = samples["sample_ori"].float()
        data_mask = samples["sample_mask"].float()
        self.optimizer.zero_grad()
        # data = data.unsqueeze(2).reshape(-1,1,seqlen)
        x, _ = self.encoder(data_mask)
        x, _ = self.token_transformer(x)
        
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.decoder(x)
        loss = self.mse(x.view(-1), data.view(-1))

        loss.backward()
        self.optimizer.step()
        return {'Total_loss': loss.item()}, \
               [self.encoder, self.token_transformer, self.projection_head]
class simmtm_channel_independent(torch.nn.Module):
    def __init__(self, head_dim, input_channel,encoder, seq_len,positive_nums=3,device='cuda:0',batch_size=256, lr = 1e-5):
        super(simmtm_channel_independent, self).__init__()
        self.encoder = encoder[0]
        self.token_transformer = encoder[1]
        self.device = device
        self.projection_head = proj_head(head_dim,256,128)
        self.awl = AutomaticWeightedLoss(2)
        self.contrastive = ContrastiveWeight(temperature=0.2,positive_nums=positive_nums)
        self.aggregation = AggregationRebuild(temperature=0.2)
        self.head = nn.Linear(100*seq_len,seq_len*input_channel)
        self.mse = torch.nn.MSELoss()
        self.network = nn.Sequential(self.encoder, self.token_transformer, self.projection_head,self.head)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=3e-4,
            betas=(0.9, 0.99)
        )
    def return_init(self):
        return {'Total_loss': 0}, \
               [self.encoder, self.token_transformer, self.projection_head]

    def update(self, samples):
        #x_in:bs, var, seqlen
        x_in = samples["sample_ori"].float().to('cpu')
        x_ori = x_in.clone().to(self.device)
        bs, var, seq_len = x_in.shape
        data_masked_m, mask = data_transform_masked4cl(x_in, 0.5, 3, 3)
        mask = torch.cat([torch.ones(x_in.shape), mask], 0).to(self.device)
        x_in = torch.cat([x_in, data_masked_m], 0).to(self.device)
        self.optimizer.zero_grad()
        # normalization
        means = torch.sum(x_in, dim=2) / torch.sum(mask == 1, dim=2)
        means = means.unsqueeze(2).detach()
        x_in = x_in - means
        x_in = x_in.masked_fill(mask == 0, 0)
        stdev = torch.sqrt((torch.sum(x_in * x_in, dim=2)+1e-5) / (torch.sum(mask == 1, dim=2) + 1e-5))
        stdev = stdev.unsqueeze(2).detach()
        x_in /= stdev
        # channel independent
        bs_extend, var, seq_len = x_in.shape
        x_in = x_in.unsqueeze(2) # x_enc: [bs x n_vars x 1 x seq_len]
        x_in = x_in.reshape(-1, 1, seq_len) # x_enc: [(bs * n_vars) x 1 x seq_len]
        #encoder
        x,_ = self.encoder(x_in)
        x,_ = self.token_transformer(x)#bs*var,channel,seqlen
        h = x.reshape(x.shape[0], -1)
        z = self.projection_head(h)#bs*var, dimension
        #point loss
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(z)
        #rebuild
        rebuild_weight_matrix, agg_x = self.aggregation(similarity_matrix, x)#agg_x:bs*var,channel,seqlen
        #decoder
        agg_x = agg_x.reshape(bs_extend, var, -1)#agg_x:bs,var,channel*seqlen
        pred_x = self.head(agg_x.reshape(agg_x.shape[0], agg_x.shape[1], -1))#pred_x:bs,var,seqlen
        # de-Normalization
        pred_x = pred_x * (stdev[:, :, 0].unsqueeze(2).repeat(1, 1, seq_len))
        pred_x = pred_x + (means[:, :, 0].unsqueeze(2).repeat(1, 1, seq_len))
        #series loss
        pred_x = pred_x[:bs]
        loss_rb = self.mse(pred_x, x_ori.detach())
        loss = self.awl(loss_cl, loss_rb)
        loss.backward()
        self.optimizer.step()
        return {'Total_loss': loss.item()}, \
               [self.encoder, self.token_transformer, self.projection_head]

class BYOLTrainer(torch.nn.Module):
    def __init__(self, head_dim, encoder, m=0.996,device='cuda:0',batch_size=256, lr = 1e-5):
        super(simmtm_channel_independent, self).__init__()
        self.online_encoder = encoder[0]
        self.online_token_transformer = encoder[1]
        self.online_network = nn.Sequential(self.online_encoder,self.online_token_transformer)
        
        self.target_encoder = copy.deepcopy(encoder[0])
        self.target_token_transformer = copy.deepcopy(encoder[1])
        self.target_network = nn.Sequential(self.target_encoder,self.target_token_transformer)
        
        self.device = device
        self.predictor = proj_head(head_dim,256,128)
        self.m = m
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(
            list(self.online_network.parameters())+list(self.predictor.parameters),
            lr=lr,
            weight_decay=3e-4,
            betas=(0.9, 0.99)
        )
        
    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    def update(self, samples):
        aug1 = samples["transformed_samples"][0].float()
        aug2 = samples["transformed_samples"][1].float()
        self.optimizer.zero_grad()
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(aug1))
        predictions_from_view_2 = self.predictor(self.online_network(aug2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(aug1)
            targets_to_view_1 = self.target_network(aug2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        loss.backward()
        self.optimizer.step()
        return loss.mean()