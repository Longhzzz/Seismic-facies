import random
from transforms3d.axangles import axangle2mat
from scipy.interpolate import CubicSpline 
import numpy as np
import torch
import warnings
import math
warnings.filterwarnings("ignore")
from scipy import interpolate
import torch.nn.functional as F
from einops import rearrange
def ts_tcc_scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return torch.from_numpy(np.concatenate((ai), axis=1))
def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)
def generate_continuous_mask(x, n=5, l=0.1):
    T = x.shape[-1]
    res = np.full(T, True, dtype=bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    t = np.random.randint(T-l-2+1)
    res[t:t+l] = False
    x = x * res
    return x
def permutation(x, max_segments=8, seg_mode="random"):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[:,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)
def GenerateRandomCurves(X, sigma=0.2, knot=4):#X:ext*attr*seqlen
    xx = (np.ones((X.shape[0],1))*(np.arange(0,X.shape[-1], (X.shape[-1]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[0]))
    x_range = np.arange(X.shape[-1])
    dd = np.zeros((X.shape[0],X.shape[-1]))
    for i in range(X.shape[0]):
        cs = interpolate.splrep(xx[:,i], yy[:,i], s=0)
        cs = interpolate.splev(x_range, cs)
        dd[i,:] = cs
    return dd#ext*seqlen
def DA_MagWarp(X, sigma):
    return X * GenerateRandomCurves(X, sigma)
def DistortTimesteps(X, sigma=0.2):
    #X:ext*attr*seqlen
    tt = GenerateRandomCurves(X, sigma) 
    #tt: ext*seqlen
    tt_cum = np.cumsum(tt, axis=1)        
    t_scale = (X.shape[-1]-1)/tt_cum[:,-1]
    tt_cum = tt_cum*t_scale.reshape(-1,1)
    return tt_cum
def DA_TimeWarp(X, sigma=0.2):
    X = X.transpose(0,1)
    #X:ext*attr*seqlen
    tt_new = DistortTimesteps(X, sigma)
    #tt_new:ext*seqlen
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[-1])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_new[i,j,:] = np.interp(x_range, tt_new[i,:], X[i,j,:])
    return torch.from_numpy(X_new).transpose(0,1)
def RandSampleTimesteps(X, nSample=1000):
    tt = np.zeros((X.shape[0],nSample), dtype=int)
    tt[0,1:-1] = np.sort(np.random.randint(1,X.shape[1]-1,nSample-2))
    tt[0,-1] = X.shape[1]-1
    return tt
def DA_RandSampling(X, nSample=1000):
    tt = RandSampleTimesteps(X, nSample)
    X_new = np.zeros(X.shape)
    X_new[0,:] = np.interp(np.arange(X.shape[1]), tt[0,:], X[0,tt[0,:]])
    return torch.from_numpy(X_new)
def data_transform_masked4cl(sample, masking_ratio, lm, positive_nums=None, distribution='geometric'):
    """Masked time series in time dimension"""

    if positive_nums is None:
        positive_nums = math.ceil(1.5 / (1 - masking_ratio))


    sample_repeat = sample.repeat(positive_nums, 1, 1)

    mask = noise_mask(sample_repeat, masking_ratio, lm, distribution=distribution)
    x_masked = mask * sample_repeat

    return x_masked.squeeze(0)


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (
            1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def noise_mask(X, masking_ratio=0.25, lm=3, distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)
    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        mask = geom_noise_mask_single(X.shape[0] * X.shape[1] * X.shape[2], lm, masking_ratio)
        mask = mask.reshape(X.shape[0], X.shape[1], X.shape[2])
    elif distribution == 'masked_tail':
        mask = np.ones(X.shape, dtype=bool)
        for m in range(X.shape[0]):  # feature dimension

            keep_mask = np.zeros_like(mask[m, :], dtype=bool)
            n = math.ceil(keep_mask.shape[1] * (1 - masking_ratio))
            keep_mask[:, :n] = True
            mask[m, :] = keep_mask  # time dimension
    elif distribution == 'masked_head':
        mask = np.ones(X.shape, dtype=bool)
        for m in range(X.shape[0]):  # feature dimension

            keep_mask = np.zeros_like(mask[m, :], dtype=bool)
            n = math.ceil(keep_mask.shape[1] * masking_ratio)
            keep_mask[:, n:] = True
            mask[m, :] = keep_mask  # time dimension
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                p=(1 - masking_ratio, masking_ratio))
    return torch.tensor(mask)