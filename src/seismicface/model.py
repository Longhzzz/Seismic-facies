import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import matplotlib.pyplot as plt
import torchvision.models as models
from layers.SelfAttention_Family import *
from layers.Transformer_EncDec import *
from layers.Embed import *
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x
    
class PositionalEncoding1D(nn.Module):
    def __init__(self, embed_dim, max_seq_length=512):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(0.1)
        
        position_encoding = self.calculate_positional_encoding(embed_dim, max_seq_length)
        self.register_buffer('position_encoding', position_encoding)

    def calculate_positional_encoding(self, embed_dim, max_seq_length):
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / embed_dim))
        position_encoding = torch.zeros(max_seq_length, embed_dim,requires_grad=False)
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        position_encoding = position_encoding.unsqueeze(0)
        return position_encoding
    def forward(self, x):
        # x : batch_size*seq_len*channel
        x = x + self.position_encoding[:, :x.size(1)]
        return self.dropout(x)
    
class Token_Transformer(nn.Module):
    def __init__(self, *, seqlen, channel, ext, dim, depth, heads, mlp_dim, channels=1, dropout=0.3):
        super().__init__()
        patch_dim = seqlen
        self.nvar = channel
        self.ext = ext
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()
        pe = PositionalEncoding1D(embed_dim=dim).position_encoding.unsqueeze(2).unsqueeze(3)
        self.register_buffer('pe',pe)
        self.dropout = nn.Dropout(0.1)
    def forward(self, forward_seq):
        # forward_seq: b * nvar * ext * channel * seq
        # x = forward_seq.transpose(1, 2)
        # x = self.pe(x)
        x = forward_seq
        if len(x.shape)<5:
            x = x.unsqueeze(3)
        # print(x.shape)
        b , nvar , ext , channel , seq = x.shape
        # print(x.shape)
        x = x.transpose(1,2)# x: b , ext , nvar , channel , seq 
        m = self.patch_to_embedding(x)
        if ext != 1:
            m = m + self.pe[:,:x.shape[1]]
            m = self.dropout(m)
        m = m.reshape(b,-1,m.shape[-1])
        b, n, _ = m.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, m), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        c_t = F.normalize(c_t,dim=1)
        return c_t, x
###########################################################################################
class Token_Transformer_paper(nn.Module):
    def __init__(self, *, seqlen, channel, ext, dim, depth, heads, mlp_dim, channels=1, dropout=0.3):
        super().__init__()
        patch_dim = seqlen
        self.nvar = channel
        self.ext = ext
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.scale = torch.cat([torch.ones(1), torch.tensor([2,2,1,1,1,2,1,1,1]).unsqueeze(-1).repeat(1,8).reshape(-1)]).reshape(1,1,1,-1)
        self.transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=0.1,
                                      output_attention=True), dim, heads),
                    dim,
                    mlp_dim,
                    dropout=0.1,
                    activation='gelu'
                ) for l in range(depth)
            ],
            norm_layer=torch.nn.LayerNorm(dim))
        
        self.to_c_token = nn.Identity()
        pe = PositionalEncoding1D(embed_dim=dim).position_encoding.unsqueeze(2).unsqueeze(3)
        self.register_buffer('pe',pe)
        self.dropout = nn.Dropout(0.1)
        # self.position_emb = nn.Linear(2, dim)
        self.sim = torch.nn.CosineSimilarity(dim=-1)
        sigma = 1.5
        self.window = self.gauss_window(seqlen,sigma)
        self.window.requires_grad=False
    def gauss_window(self,w,sigma):
        # Create a tensor for the window
        window = torch.arange(w).float()
        # Calculate the Gaussian function
        window = torch.exp(-0.5 * ((window - (w) // 2) / sigma) ** 2)
        # Normalize the window
        window /= window.sum()
        return window * w
    def forward(self, forward_seq, epoch=50,attn_mask=None,):
         # forward_seq: b * nvar * ext * channel/None * seq
        # x = self.pe(x)
        x = forward_seq * self.window.to(forward_seq.device)
        if len(x.shape)<5:
            x = x.unsqueeze(3)
        # print(x.shape)
        x = torch.mean(x, dim=2, keepdim=True)
        bs , nvar , ext , channel , seq = x.shape
        # print(x.shape)
        x = x.transpose(1,2)# x: b , ext , nvar , channel , seq 
        m = self.patch_to_embedding(x)
        # if ext == 999:
        #     m = m + self.pe[:,:x.shape[1]]
        #     m = self.dropout(m)
        
        # m = m.reshape(bs,-1,m.shape[-1])#see ext as attrs
        m = m.reshape(bs,-1,m.shape[-1])#see ext as one sample,use smooth loss
        
        b, n, _ = m.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, m), dim=1)
        x,attn = self.transformer(x,attn_mask)
        c_t = self.to_c_token(x[:, 0])
        c_t = F.normalize(c_t,dim=1)
        c_t = c_t.reshape(bs,ext,-1) # use smooth loss 
        if ext > 1 and epoch > 10:
            weight = self.sim(c_t.unsqueeze(1),c_t.unsqueeze(2))[:,int((ext-1)/2)] ** 0
            weight[:,int((ext-1)/2)] = 0
            weight = weight/weight.sum(-1,True)
            r_t = (c_t * weight.unsqueeze(2)).sum(1)
            a = 0.5
            c_t = a * c_t[:,int((ext-1)/2)] + (1-a) * r_t
            c_t = F.normalize(c_t,dim=-1)
        else:
            c_t = c_t[:,int((ext-1)/2)]
        
        return c_t, attn # use smooth loss
        # return c_t, c_t # see ext as attrs
#######################################################################################
class cnn1d_depthwise_block(nn.Module):
    def __init__(self,input_channels,final_out_channels=128, stride=1,dropout=0.3,groups=1):
        super(cnn1d_depthwise_block, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, final_out_channels, kernel_size=3,
                      stride=stride, bias=True, padding='same',groups=groups),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(final_out_channels, final_out_channels, kernel_size=3,
                      stride=stride, bias=True, padding='same',groups=groups),
            # nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            # # nn.MaxPool1d(kernel_size=mp, stride=mp, padding=mp//2),
            # nn.Dropout(dropout)
        )
    def forward(self, x, mode='predict'):
        bs,nvar,ext,seqlen = x.shape
        
        x = x.reshape(bs,-1,seqlen)
        x = self.conv_block1(x)#bs,nvar*channel,seqlen
        # x = self.conv_block2(x)
        x = x.unsqueeze(1)
        ext = 1
        x = x.reshape(bs,ext,-1,seqlen)
        x = x.transpose(1,2)
        ###before######################
        # x = x.transpose(1,2)
        # x = x.reshape(-1,nvar,seqlen)
        # x = self.conv_block1(x)
        # # x = self.conv_block2(x)
        # x = x.reshape(bs,ext,-1,seqlen)
        # x = x.transpose(1,2)
        #################################
        # x : bs,nvar*channel,ext,seqlen
        x = x.reshape(bs,nvar,-1,ext,seqlen)
        # x : bs,nvar,channel,ext,seqlen
        if mode == 'train':
            n0 = torch.randint(3,nvar+1,size=(1,)).item()
            x = x[:,torch.randperm(nvar)[:n0].tolist()]
        x = x.reshape(bs,-1,ext,seqlen)#bs,nvar*channel,ext,seqlen
        return x ,x
#############################################################################################
class Token_TransformerDouble(nn.Module):
    def __init__(self, *, seqlen,attrs, dim, depth, heads, mlp_dim, channels=1, dropout=0.3):
        super().__init__()
        self.patch_to_embedding = nn.Linear(seqlen, dim)
        self.ext_to_embedding = nn.Linear(dim, dim)
        self.patch_c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.ext_c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.patch_transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.ext_transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.attrs = attrs
        self.to_c_token = nn.Identity()
        self.ext_pe = PositionalEncoding1D(embed_dim=dim)
    def forward(self, forward_seq):
        # forward_seq: b * channel * seq
        b ,channel, seq = forward_seq.shape
        x = forward_seq.reshape(b,self.attrs,-1,seq)
        # x: b*attrs*ext*seq
        b,attrs,ext,seq = x.shape
        x = x.transpose(1,2)
        # x: b*ext*attrs*seq
        x = x.reshape(-1,self.attrs,seq)
        # x: (b*ext)*attrs*seq
        m = self.patch_to_embedding(x)
        # m: (b*ext)*attrs*dim
        patch_c_tokens = repeat(self.patch_c_token, '() n d -> b n d', b=b*ext)
        x = torch.cat((patch_c_tokens, m), dim=1)
        x = self.patch_transformer(x)
        c_t = self.to_c_token(x[:, 0])
        #c_t (b*ext)*dim
        
        c_t = c_t.reshape(b,ext,-1)
        #c_t b*ext+1*dim
        c_t = self.ext_to_embedding(c_t)
        ext_c_tokens = repeat(self.ext_c_token, '() n d -> b n d', b=b)
        c_t = torch.cat((ext_c_tokens, c_t), dim=1)
        c_t = self.ext_pe(c_t)
        x = self.ext_transformer(c_t)
        #c_t b*ext+1*dim
        c_t = self.to_c_token(x[:, 0])
        #c_t b*dim
        
        return c_t, x
class Token_TransformerT(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.3):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()
        self.pe = PositionalEncoding1D(embed_dim=dim)
    def forward(self, forward_seq):
        # forward_seq: b * nvar * ext * channel * seq
        # x = forward_seq.reshape(forward_seq.shape[0],-1,forward_seq.shape[-1])
        x = forward_seq
        x = x.reshape(x.shape[0],-1,x.shape[-1])
        x = x.transpose(1, 2)
        m = self.patch_to_embedding(x)
        b, n, _ = m.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, m), dim=1)
        x = self.pe(x)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        c_t = F.normalize(c_t,dim=1)
        return c_t, x
class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.3):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()
        self.pe = PositionalEncoding1D(embed_dim=patch_size)
    def forward(self, forward_seq):
        x = forward_seq.transpose(1, 2)
        x = self.pe(x)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        x = self.transformer(x)
        c_t = x
        c_t = c_t.transpose(1, 2)
        return c_t, x
class Mid_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.3):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()
        self.pe = PositionalEncoding1D(embed_dim=32)
    def forward(self, forward_seq):
        #forward_seq: b*seq*channel
        x = self.patch_to_embedding(forward_seq)
        x = self.pe(x)
        b, n, _ = x.shape
        x = self.transformer(x)
        c_t = x
        c_t = c_t[:,int((x.shape[1]-1)/2),:]
        return c_t, x
class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len*d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x seq_len x d_model]
        x = self.flatten(x) # [bs x n_vars x (seq_len * d_model)]
        x = self.linear(x) # [bs x n_vars x seq_len]
        x = self.dropout(x) # [bs x n_vars x seq_len]
        return x

class Pooler_Head(nn.Module):
    def __init__(self, seq_len, d_model, head_dropout=0):
        super().__init__()

        pn = seq_len * d_model
        dimension = 128
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, pn // 2),
            nn.BatchNorm1d(pn // 2),
            nn.ReLU(),
            nn.Linear(pn // 2, dimension),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):  # [(bs * n_vars) x seq_len x d_model]
        x = self.pooler(x) # [(bs * n_vars) x dimension]
        return x
class cnn1d_fe(nn.Module):
    def __init__(self, input_channels, mid_channels, final_out_channels, stride, dropout=0.35,mp=1):
        super(cnn1d_fe, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, mid_channels, kernel_size=3,
                      stride=stride, bias=False, padding='same'),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=mp, stride=mp, padding=mp//2),
            nn.Dropout(dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels * 2, kernel_size=3,
                      stride=1, bias=False, padding='same'),
            nn.BatchNorm1d(mid_channels * 2),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=mp, stride=mp, padding=mp//2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(mid_channels , final_out_channels, kernel_size=3,
                      stride=1, bias=False,padding='same'),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=mp, stride=mp, padding=mp//2),
        )
    def forward(self, x_in):
        x = self.conv_block1(x_in)
        # x = self.conv_block2(x)
        x = self.conv_block3(x)
        # x = F.normalize(x, dim=1)
        return x, x

class proj_head(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim,bn =True):
        super(proj_head, self).__init__()
        if bn:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, output_dim),
            )
    def forward(self, x):
        x = self.classifier(x)
        return x
class origin(nn.Module):
    def __init__(self):
        super(origin, self).__init__()
        self.head = nn.Identity()
    def forward(self, x):
        x = self.head(x)
        return x ,x
class cnn1d_block(nn.Module):
    def __init__(self,input_channels,final_out_channels=128, stride=1,dropout=0.3):
        super(cnn1d_block, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, final_out_channels, kernel_size=3,
                      stride=stride, bias=False, padding='same'),
            # nn.BatchNorm1d(final_out_channels),
            # nn.ReLU(),
            # # nn.MaxPool1d(kernel_size=mp, stride=mp, padding=mp//2),
            # nn.Dropout(dropout)
        )
    def forward(self, x):
        x = self.conv_block1(x)
        return x ,x
class cnn1d_channel_independent_block(nn.Module):
    def __init__(self,input_channels,final_out_channels=128, stride=1,dropout=0.25):
        super(cnn1d_channel_independent_block, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, final_out_channels//2, kernel_size=3,
                      stride=stride, bias=False, padding='same'),
            # nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            # # nn.MaxPool1d(kernel_size=mp, stride=mp, padding=mp//2),
            nn.Dropout(dropout),
            nn.Conv1d(final_out_channels//2, final_out_channels, kernel_size=3,
                      stride=stride, bias=False, padding='same')
        )
    def forward(self, x):
        
        b,nvar,ext,seqlen = x.shape
        x = x.reshape(-1, 1, seqlen)
        x = self.conv_block1(x)
        x = x.reshape(b,nvar,ext,-1,seqlen)
        # x = x.reshape(b,-1,seqlen)
        return x ,x
class cnn2d_channel_independent_block(nn.Module):
    def __init__(self,input_channels,final_out_channels=128, stride=1,dropout=0.25,ext=1):
        super(cnn2d_channel_independent_block, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, final_out_channels//2, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(final_out_channels//2),
            nn.ReLU(),
            # # nn.MaxPool1d(kernel_size=mp, stride=mp, padding=mp//2),
            nn.Dropout(dropout),
            nn.Conv2d(final_out_channels//2, final_out_channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.l = torch.sqrt(torch.tensor(ext)).int()
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Linear(512,final_out_channels)
        self.model=nn.Sequential(nn.Conv2d(1, 3, kernel_size=1, padding=0, stride=1,bias=False),
                    resnet)
        

    def forward(self, x):
        b,nvar,ext,seqlen = x.shape
        x = x.transpose(2,3)
        x = x.reshape(b*nvar*seqlen,1,ext)
        x = x.reshape(b*nvar*seqlen,1,self.l,self.l)
        x = self.conv_block1(x)
        # print(x.shape)
        x = x.reshape(b,nvar,seqlen,-1)
        x = x.transpose(2,3)
        x = x.unsqueeze(2)
        return x ,x