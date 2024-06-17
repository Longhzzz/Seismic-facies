import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from model import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from einops import rearrange
from loss import NTXentLoss,  Entropy
from layers.SelfAttention_Family import *
from layers.Transformer_EncDec import *
from layers.Embed import *
def ts_tcc_encoder(input_channel,seqlen=0):
    encoder = cnn1d_fe(input_channels=input_channel, mid_channels=32, final_out_channels=128, stride=1)
    token_transformer = Token_Transformer(patch_size=128, dim=100, depth=4, heads=4, mlp_dim=64)
    return encoder,token_transformer
def baseline_encoder(input_channel,seqlen=0):
    encoder = cnn1d_fe(input_channels=input_channel, mid_channels=32, final_out_channels=128, stride=1)
    token_transformer = Seq_Transformer(patch_size=128, dim=100, depth=4, heads=4, mlp_dim=64)
    return encoder,token_transformer
def cnn1d_encoder(input_channel,seqlen=0):
    encoder = cnn1d_fe(input_channels=input_channel, mid_channels=32, final_out_channels=128, stride=1,mp=2)
    token_transformer = origin()
    return encoder, token_transformer
#############################################################################
def transformer_encoder(input_channel,seqlen=0,ext=0):
    # encoder = cnn1d_channel_independent_block(input_channels=input_channel,final_out_channels=16, stride=1,dropout=0.1)
    # encoder = cnn2d_channel_independent_block(input_channels=input_channel,final_out_channels=16, stride=1,dropout=0.3,ext=ext)
    # encoder = origin()
    encoder = cnn1d_depthwise_block(input_channels=input_channel*ext,final_out_channels=input_channel*8, groups=input_channel, stride=1,dropout=0.2)
    token_transformer = Token_Transformer_paper(seqlen=seqlen, channel=input_channel,ext=ext,dim=64, depth=2, heads=8, mlp_dim=256)
    return encoder,token_transformer
##############################################################################
def cnn2d_transformer_encoder(input_channel,seqlen=0,ext=0):
    encoder = cnn2d_channel_independent_block(input_channels=input_channel,final_out_channels=32, stride=1,dropout=0.3,ext=ext)
    token_transformer = Token_Transformer(patch_size=seqlen, channel=input_channel,ext=ext,dim=64, depth=2, heads=2, mlp_dim=128)
    return encoder,token_transformer
def cnn2d_transformerT_encoder(input_channel,seqlen=0,ext=0):
    encoder = cnn2d_channel_independent_block(input_channels=input_channel,final_out_channels=32, stride=1,dropout=0.3,ext=ext)
    token_transformer = Token_TransformerT(patch_size=seqlen, channel=input_channel,ext=ext,dim=64, depth=2, heads=2, mlp_dim=128)
    return encoder,token_transformer
def transformerDouble_encoder(input_channel,seqlen=0):
    encoder = origin()
    token_transformer = Token_TransformerDouble(seqlen=seqlen,attrs=input_channel, dim=32, depth=2, heads=2, mlp_dim=128)
    return encoder,token_transformer
def transformerT_encoder(input_channel,seqlen=0,ext=1):
    encoder = cnn1d_channel_independent_block(input_channels=input_channel,final_out_channels=32, stride=1,dropout=0.3)
    token_transformer = Token_TransformerT(patch_size=1*ext*32, dim=64, depth=2, heads=2, mlp_dim=128)
    return encoder,token_transformer
def cnn_transformer_encoder(input_channel,seqlen=0):
    encoder = cnn1d_fe(input_channels=input_channel, mid_channels=16, final_out_channels=32, stride=1)
    token_transformer = Token_Transformer(patch_size=seqlen, dim=32, depth=4, heads=2, mlp_dim=64)
    return encoder,token_transformer
def Mid_transformer_encoder(input_channel,seqlen=0):
    encoder = origin()
    token_transformer = Mid_Transformer(patch_size=input_channel, dim=32, depth=2, heads=2, mlp_dim=64)
    return encoder,token_transformer
def simmtm_encoder(input_channel,seqlen=0):
    # Embedding
    encoder = origin()
    # Encoder
    token_transformer = nn.Sequential(
        DataEmbedding(1, d_model=128, embed_type='fixed', freq='h', dropout=0.3),
        Encoder([EncoderLayer(AttentionLayer(
            DSAttention(False, factor=1, attention_dropout=0.3,output_attention=False), 
            d_model=128, n_heads=4),d_model=128,d_ff=256,dropout=0.3,activation=F.relu) for l in range(4)],
        norm_layer=torch.nn.LayerNorm(128),)
    )
    return encoder,token_transformer