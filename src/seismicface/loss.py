import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns 
import matplotlib.pyplot as plt
import copy
from torch import Tensor
import warnings
import math
class MSNLoss(nn.Module):
    
    def __init__(
        self,temperature=0.1,c='None',sinkhorn_iterations=0,regularization_weight=1.0,target_distribution=None,power_law_exponent=2):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be in (0, inf) but is {temperature}.")
        if sinkhorn_iterations < 0:
            raise ValueError(
                f"sinkhorn_iterations must be >= 0 but is {sinkhorn_iterations}."
            )

        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.regularization_weight = regularization_weight
        self.target_distribution = target_distribution
        self.power_law_exponent = power_law_exponent
        self.c = c

    def forward(self,anchors,targets,prototypes,step,epoch,path,target_sharpen_temperature= 0.25,):
        num_views = anchors.shape[0] // targets.shape[0]
        anchors = F.normalize(anchors, dim=1)
        targets = F.normalize(targets, dim=1)
        prototypes = F.normalize(prototypes, dim=1)

        # anchor predictions
        anchor_probs = self.prototype_probabilities(
            anchors, prototypes, temperature=self.temperature
        )

        # target predictions
        with torch.no_grad():
            target_probs = self.prototype_probabilities(
                targets, prototypes, temperature=self.temperature
            )
            target_probs = self.sharpen(target_probs, temperature=target_sharpen_temperature)
            if self.sinkhorn_iterations > 0:
                target_probs = self.sinkhorn(
                    probabilities=target_probs,
                    iterations=self.sinkhorn_iterations,
                    gather_distributed=self.gather_distributed,
                )
            target_probs = target_probs.repeat((num_views, 1))

        # cross entropy loss
        loss = torch.mean(torch.sum(torch.log(anchor_probs ** (-target_probs)), dim=1))

        # regularization loss
        if self.regularization_weight > 0:
            mean_anchor_probs = torch.mean(anchor_probs, dim=0)
            reg_loss = self.regularization_loss(mean_anchor_probs=mean_anchor_probs,c=self.c)
            loss += self.regularization_weight * reg_loss
        return loss
    def prototype_probabilities(self,queries,prototypes,temperature,) :
        return F.softmax(torch.matmul(queries, prototypes.T) / temperature, dim=1)
    def sharpen(self,probabilities, temperature):
        probabilities = probabilities ** (1.0 / temperature)
        probabilities /= torch.sum(probabilities, dim=1, keepdim=True)
        return probabilities
    def regularization_loss(self, mean_anchor_probs,c='None'):
        if c == 'None':
            loss = -torch.sum(torch.log(mean_anchor_probs ** (-mean_anchor_probs)))
            loss += math.log(float(len(mean_anchor_probs)))
        elif c == 'power':
            power_dist = self._power_law_distribution(size=mean_anchor_probs.shape[0],exponent=self.power_law_exponent,device=mean_anchor_probs.device,)
            loss = F.kl_div(input=mean_anchor_probs.log(), target=power_dist, reduction="sum")
        else:
            target_dist = self.target_distribution.to(mean_anchor_probs.device)
            loss = F.kl_div(input=mean_anchor_probs.log(), target=target_dist, reduction="sum")
        return loss
    def _power_law_distribution(self,size, exponent, device) :
        """Returns a power law distribution summing up to 1."""
        k = torch.arange(1, size + 1, device=device)
        power_dist = k ** (-exponent)
        power_dist = power_dist / power_dist.sum()
        return power_dist
    
class DCLW(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity, shape):
        super(DCLW, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.shape = shape
        self.softmax = torch.nn.Softmax(dim=-1)
        self.maxd = torch.sqrt(torch.tensor(shape[0])**2+torch.tensor(shape[1])**2).to(device)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.softmax = torch.nn.Softmax()
        self.sigma = 0.05
        # self.back = np.fromfile('log4/jintao/predict/kmeans/cnn1d_encoder/simple_freezy_w10_s0_addpos/ssl/ts_tcc/cnn1d_encoder/Warp_aug+jitter_w10_s0_addpos/60/fe+te/Kmeans_6.dat',dtype=np.float32).reshape(shape)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity
    def cal_distance(self,position):
        diff = position.unsqueeze(1) - position.unsqueeze(0)
        distance_matrix = torch.sqrt(torch.sum(diff**2, dim=-1))
        return distance_matrix
    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, position,step,epoch,path):
        representations = torch.cat([zjs, zis], dim=0)
        w_matrix = self.cal_distance(position[:,0:2])
        # w_matrix = w_matrix + torch.eye(position.shape[0]).to(self.device)
        # w_matrix[w_matrix>0.1] = 1
        # w_matrix = self.cal_distance(position)
        w_matrix_neg = 1 + 1 / (1e-8 + 0.001 * w_matrix)
        w_matrix_neg = w_matrix_neg - 1e16 * torch.eye(self.batch_size).to(self.device)
        
        similarity_matrix = self.similarity_function(representations, representations) 
        topk = 0
        w_sim = similarity_matrix[-self.batch_size:,-self.batch_size:] * w_matrix_neg
        v,k_w = w_sim.topk(topk, dim=1, largest=True, sorted=False)
        v,k = similarity_matrix[-self.batch_size:,-self.batch_size:].topk(topk, dim=1, largest=True, sorted=False)
        top_mask = torch.zeros(w_sim.shape).to(self.device)
        top_mask.scatter_(1,k_w,1)
        top_mask = top_mask.repeat(2,2)
        
        if step == 0 and (epoch%5 == 0 or epoch == 1):
            pass
            # p0 = position[0:1].clone().detach().cpu()
            # p_w = position[k_w[0]].clone().detach().cpu()
            # p = position[k[0]].clone().detach().cpu()
            # plt.cla()
            # plt.figure(figsize=(10,3))
            # # plt.imshow(self.back,cmap='gray')
            # plt.imshow(np.zeros(self.shape))
            # plt.scatter(p_w[:,1],p_w[:,0],s=10,c='r')
            # plt.scatter(p0[:,1],p0[:,0],s=10,c='y')
            # plt.savefig(path+'/png/Wtop{}_{}.png'.format(topk,epoch),dpi=200)
            # plt.cla()
            # plt.figure(figsize=(10,3))
            # # plt.imshow(self.back,cmap='gray')
            # plt.imshow(np.zeros(self.shape))
            # plt.scatter(p[:,1],p[:,0],s=10,c='r')
            # plt.scatter(p0[:,1],p0[:,0],s=10,c='y')
            # plt.savefig(path+'/png/top{}_{}.png'.format(topk,epoch),dpi=200)
            # plt.cla()
            
        w_matrix_posi = 1 - 1 / (1 + 0.003 * w_matrix)
        w_matrix_posi = self.batch_size * w_matrix_posi / w_matrix_posi.sum(1,True)
        w_matrix_posi = w_matrix_posi + torch.eye(position.shape[0]).to(self.device)
        w_matrix_posi = w_matrix_posi.repeat(2,2)
        
        similarity_matrix = similarity_matrix *1#  w_matrix_posi
        mask_l = torch.from_numpy(np.eye(self.batch_size*2,k=self.batch_size)).to(self.device)
        mask_r = torch.from_numpy(np.eye(self.batch_size*2,k=-self.batch_size)).to(self.device)
        mask_diag = torch.from_numpy(np.eye(self.batch_size*2)).to(self.device)
        
        
        # l_pos = similarity_matrix[mask_l.bool()]
        # r_pos = similarity_matrix[mask_r.bool()]
        # positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, -1)
        positives = similarity_matrix[(mask_l+mask_r+top_mask).bool()].view(2 * self.batch_size, -1) * 1
        
        negatives = similarity_matrix[(1-(mask_l+mask_r+mask_diag+top_mask)).bool()].view(2 * self.batch_size, -1)
        #WDCL
        # weight_l = 2 - self.batch_size * self.softmax(l_pos*self.temperature/self.sigma)
        # weight_r = 2 - self.batch_size * self.softmax(r_pos*self.temperature/self.sigma)
        # weight = torch.cat([weight_l, weight_r]).view(2 * self.batch_size, 1)
        logits = torch.cat((positives, negatives), dim=1)/ self.temperature
        # print(logits[:10]*self.temperature)
        logits = self.softmax(logits)
        loss = -torch.log(logits[:,:positives.shape[1]]).mean(1).sum()
        return loss / (2 * self.batch_size)
    
class W_NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity, shape):
        super(W_NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.maxd = torch.sqrt(torch.tensor(shape[0])**2+torch.tensor(shape[1])**2).to(device)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.softmax = torch.nn.Softmax()

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity
    def cal_distance(self,position):
        diff = position.unsqueeze(1) - position.unsqueeze(0)
        distance_matrix = torch.sqrt(torch.sum(diff**2, dim=-1))
        return distance_matrix
    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, position,step,epoch,path):
        representations = torch.cat([zjs, zis], dim=0)
        w_matrix = self.cal_distance(position)/self.maxd 
        
        w_matrix = w_matrix + torch.eye(position.shape[0]).to(self.device)
        # w_matrix = 1/(1+0.1*w_matrix)
        # print('W: ',w_matrix)
        w_matrix = w_matrix.repeat(2,2)
        similarity_matrix = self.similarity_function(representations, representations)
        
        similarity_matrix = similarity_matrix * w_matrix
        # if step == 0:
        #     similarity_matrix_copy = similarity_matrix.detach().clone().cpu()
        #     plt.cla()
        #     sns.distplot((similarity_matrix_copy )[:512,:1024])
        #     plt.savefig('dist_proj.png')
        #     plt.cla()
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        
        logits = self.softmax(logits)
        logits = -torch.log(logits)
        
        loss = torch.sum(logits[:,0])
        return loss / (2 * self.batch_size)
class W_NTXentLoss_l(torch.nn.Module):

    def __init__(self, device, temperature, use_cosine_similarity, shape):
        super(W_NTXentLoss_l, self).__init__()
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.maxd = torch.sqrt(torch.tensor(shape[0])**2+torch.tensor(shape[1])**2).to(device)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.softmax = torch.nn.Softmax()
    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity
    def cal_distance(self,position):
        diff = position.unsqueeze(1) - position.unsqueeze(0)
        distance_matrix = torch.sqrt(torch.sum(diff**2, dim=-1))
        return distance_matrix/self.maxd 

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, z,label,position):
        representations = z # b * dim
        w_matrix = self.cal_distance(position)
        
        w_matrix = w_matrix + torch.eye(position.shape[0]).to(self.device)
        # print('W: ',w_matrix)
        similarity_matrix = self.similarity_function(representations, representations)
        # b*b
        similarity_matrix = similarity_matrix #* w_matrix

        mask = label.unsqueeze(0) - label.unsqueeze(1)
        mask = (mask == 0)
        mask = mask[(1-torch.eye(label.shape[0])).bool()].view(label.shape[0],-1)
        similarity_matrix = similarity_matrix[(1-torch.eye(label.shape[0])).bool()].view(label.shape[0],-1)
        
        logits = similarity_matrix
        logits /= self.temperature

        logits = self.softmax(logits)
        logits = -torch.log(logits)
        loss = torch.sum(logits * mask,dim = -1)
        loss = loss / torch.sum(mask,dim=-1)
        loss = torch.sum(loss)

        return loss / label.shape[0]

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs,a,b,c):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)
        
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        # print(logits)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy
class AutomaticWeightedLoss(torch.nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class ContrastiveWeight(torch.nn.Module):

    def __init__(self, temperature,positive_nums):
        super(ContrastiveWeight, self).__init__()
        self.temperature = temperature

        self.bce = torch.nn.BCELoss()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.positive_nums = positive_nums

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size):
        diag = np.eye(cur_batch_size)
        mask = torch.from_numpy(diag)
        mask = mask.type(torch.bool)#bs*bs

        oral_batch_size = cur_batch_size // (self.positive_nums + 1)

        positives_mask = np.zeros(similarity_matrix.size())#bs*bs
        for i in range(self.positive_nums + 1):
            ll = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size * i)
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size * i)
            positives_mask += ll
            positives_mask += lr

        positives_mask = torch.from_numpy(positives_mask).to(similarity_matrix.device)
        positives_mask[mask] = 0

        negatives_mask = 1 - positives_mask
        negatives_mask[mask] = 0

        return positives_mask.type(torch.bool), negatives_mask.type(torch.bool)

    def forward(self, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape

        # get similarity matrix among mask samples
        norm_emb = F.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))

        # get positives and negatives similarity
        positives_mask, negatives_mask = self.get_positive_and_negative_mask(similarity_matrix, cur_batch_shape[0])

        positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)

        # generate predict and target probability distributions matrix
        logits = torch.cat((positives, negatives), dim=-1)
        y_true = torch.cat(
            (torch.ones(cur_batch_shape[0], positives.shape[-1]), torch.zeros(cur_batch_shape[0], negatives.shape[-1])),
            dim=-1).to(batch_emb_om.device).float()

        # multiple positives - KL divergence
        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)

        return loss, similarity_matrix, logits, positives_mask


class AggregationRebuild(torch.nn.Module):

    def __init__(self, temperature):
        super(AggregationRebuild, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mse = torch.nn.MSELoss()

    def forward(self, similarity_matrix, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape

        # get the weight among (oral, oral's masks, others, others' masks)
        similarity_matrix /= self.temperature

        similarity_matrix = similarity_matrix - torch.eye(cur_batch_shape[0]).to(
            similarity_matrix.device).float() * 1e12
        rebuild_weight_matrix = self.softmax(similarity_matrix)

        batch_emb_om = batch_emb_om.reshape(cur_batch_shape[0], -1)

        # generate the rebuilt batch embedding (oral, others, oral's masks, others' masks)
        rebuild_batch_emb = torch.matmul(rebuild_weight_matrix, batch_emb_om)

        # get oral' rebuilt batch embedding
        rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1], -1)

        return rebuild_weight_matrix, rebuild_oral_batch_emb