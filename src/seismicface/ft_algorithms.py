import copy
import random
import torch
import torch.nn as nn
import numpy as np
import os
import warnings
from model import *
warnings.simplefilter(action='ignore', category=FutureWarning)
from einops import rearrange
from loss import NTXentLoss,  Entropy
from layers.SelfAttention_Family import *
from layers.Transformer_EncDec import *
from layers.Embed import *
import torch.nn.functional as F

class simple_classifier(torch.nn.Module):
    def __init__(self, encoder, n_class, in_dim,device='cuda:0',batch_size=256, lr = 1e-6):
        super(simple_classifier, self).__init__()
        self.encoder = encoder[0]
        self.token_transformer = encoder[1]
        
        self.projection_head = proj_head(in_dim,64,n_class)
        self.network = nn.Sequential(self.encoder, self.token_transformer, self.projection_head)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=3e-4,
            betas=(0.9, 0.99)
        )
        self.flatten = nn.Flatten()
        self.cross_entropy = nn.CrossEntropyLoss(torch.tensor([1,3,7]).float())
    def return_init(self):
        return {'Total_loss': 0}, \
               [self.encoder, self.token_transformer, self.projection_head]

    def update(self, samples):
        data = samples["sample_ori"].float()
        label = samples["label"].long()
        self.optimizer.zero_grad()

        x, _ = self.encoder(data)
        x, _ = self.token_transformer(x)
        x = self.flatten(x)
        x = self.projection_head(x)
        loss = self.cross_entropy(x, label) 
        loss.backward()
        self.optimizer.step()
        return {'Total_loss': loss.item()}, \
               [self.encoder, self.token_transformer, self.projection_head]