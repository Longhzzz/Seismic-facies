from model import *
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,MiniBatchKMeans
import matplotlib.pyplot as plt
import os
from cigvis import colormap
import seaborn as sns 
import random
from sklearn.manifold import TSNE
from kmeans_pytorch import kmeans as Kmeans_torch
# from kmeans_gpu import KMeans

def classifier(model,train_dl,device,matshape,save_path,pos=None):
    encoder = model[0]
    token_transformer = model[1]
    head = model[2]
    encoder.eval()
    token_transformer.eval()
    head.eval()
    mat = np.zeros((matshape[0], matshape[1]))-1
    with torch.no_grad():
        for data in tqdm(train_dl):
            sample = data['sample_ori'].float().to(device)
            posi = data['position'].float()
            x,_ = encoder(sample)
            x,_ = token_transformer(x)
            x = x.reshape(x.shape[0],-1)
            x = head(x)
            pred = x.detach().argmax(dim=1).to('cpu').numpy()
            posi = posi.numpy().astype(np.int32)
            mat[posi[:,0],posi[:,1]] = pred
    plt.figure(figsize=(10, 10))
    if pos is not None:
        plt.matshow(mat.T, origin='lower',cmap='jet',alpha=0.7)
        ax = plt.scatter(pos[:,0],pos[:,1],c=pos[:,3],s=5,cmap='hot')
        plt.colorbar(ax)
        
    else:
        plt.matshow(mat.T, origin='lower',cmap='jet')
    plt.savefig(os.path.join(save_path,'classifier.png'), dpi = 400)
    mat.astype(np.float32).tofile(os.path.join(save_path,'classifier.dat'))
    
    values = [0.0,3.0]
    colors = ['blue','blue']
    new_cmap = colormap.custom_disc_cmap(values,colors)
    fg_cmap = colormap.set_alpha_except_min(new_cmap,1,False)
    for i in np.unique(mat):
        plt.figure(figsize=(10, 10))
        mat_c = (mat==i).astype(np.float32)
        if pos is not None:
            ax = plt.matshow(mat_c.T,origin='lower', cmap=fg_cmap,alpha=0.7)
            ax = plt.scatter(pos[:,0],pos[:,1],c=pos[:,3],s=5,cmap='hot')
            plt.colorbar(ax)
            
        else:
            plt.matshow(mat_c.T, origin='lower',cmap='jet')
        plt.savefig(os.path.join(save_path,'classifier_{}c.png'.format(i)), dpi = 400)
        # mat_c.astype(np.float32).tofile(os.path.join(save_path,'classifier_{}c.dat'.format(i)))
def cls(model,cls, train_dl,device,matshape,save_path,pos = None):
    #transformer features
    encoder = model[0]
    token_transformer = model[1]
    encoder.eval()
    token_transformer.eval()
    cls.eval()
    mat = np.zeros((matshape[0], matshape[1]))-1
    with torch.no_grad():
        for data in tqdm(train_dl):
            sample = data['sample_ori'].float().to(device)
            posi = data['position'].float()
            #sample: b*channel*seq
            x,_ = encoder(sample)
            x,_ = token_transformer(x)
            pred = torch.argmax(cls(x).detach())
            mat[posi[:,0].int(),posi[:,1].int()] = pred.to('cpu').numpy()
    plt.matshow(mat, cmap='jet')
    plt.savefig(os.path.join(save_path,'cls.png'), dpi = 400)
            
# def kmeans(model,train_dl,device,matshape,save_path,pos = None):
#     #transformer features
#     features = []
#     position = []
#     encoder = model[0]
#     token_transformer = model[1]
#     encoder.eval()
#     token_transformer.eval()
#     with torch.no_grad():
#         for data in tqdm(train_dl):
#             sample = data['sample_ori'].float().to(device)
#             posi = data['position'].float()
#             #sample: b*channel*seq
#             x,_ = encoder(sample)
#             x,_ = token_transformer(x)
#             # x: b * seq * dim/b * dim
#             if len(x.shape)>2 and (x.shape[1]>64):
#                 x_flat = torch.max(x, dim=-1)[0]
#             else:
#                 x_flat = x.reshape(x.shape[0],-1)
#             features.append(x_flat.to('cpu').numpy().tolist())
#             position.append(posi.numpy().tolist())
#     print('output feature shape:', len(features)*len(features[0]), len(features[0][0]))
#     features = np.concatenate(features,axis=0)
#     position = np.concatenate(position,axis=0).astype(np.int32)
#     print('draw feature shape:', features.shape)
    
#     pca = PCA(n_components=0.9)
#     if features.shape[1]>500:
#         features = pca.fit_transform(features)
#     #construct tsne
#     tsne = TSNE(n_components=2)
#     k = pca.fit_transform(features[:4196])
#     print('k shape:', k.shape)
#     X_tsne = tsne.fit_transform(k)
#     # plot similarity distribute
#     plt.cla()
#     sim = torch.nn.CosineSimilarity(dim=-1)
#     index = random.sample(range(0,features.shape[0]),4196)
#     sim_m = sim(torch.tensor(features[index,:]).unsqueeze(1), torch.tensor(features[index,:]).unsqueeze(0))
#     sns.distplot(sim_m.reshape(-1))
#     plt.xlim([-1,1])
#     plt.savefig(os.path.join(save_path,'dist.png'),dpi=400)
#     plt.cla()
    
#     # plot kmeans
#     mat = np.zeros((matshape[0], matshape[1]))-1
#     for clus in range(3,10):
#         print('Kmeans_{}_model_{}.png'.format(clus,len(model)))
#         #kmeans = MiniBatchKMeans(init='k-means++', n_clusters=clus, batch_size=10000, random_state=42, verbose=0).fit(emb_num)
#         kmeans = KMeans(init='k-means++', n_clusters=clus, random_state=42,
#                         verbose=0).fit(features)
#         mat[position[:,0],position[:,1]] = kmeans.labels_

#         #plot tsne
#         plt.cla()
#         plt.figure(figsize=(10, 10))
#         plt.scatter(X_tsne[:,0], X_tsne[:,1],c=kmeans.labels_[:4196])
#         plt.savefig(os.path.join(save_path,'Tsne_{}.png'.format(clus)), dpi = 400)
#         plt.cla()
        
#         #plot all
#         plt.figure(figsize=(10, 10))
#         if pos is not None:
#             ax=plt.matshow(mat,cmap='jet',alpha=0.7)
#             plt.scatter(pos[:,1],pos[:,0],c=pos[:,3],s=5,cmap='hot')
#             plt.colorbar(ax)
            
#         else:
#             plt.matshow(mat, cmap='jet')
#         plt.savefig(os.path.join(save_path,'Kmeans_{}.png'.format(clus)), dpi = 400)
#         plt.cla()
#         mat.astype(np.float32).tofile(os.path.join(save_path,'Kmeans_{}.dat'.format(clus)))
#         # #plot single
#         values = [0.0,3.0]
#         colors = ['blue','blue']
#         new_cmap = colormap.custom_disc_cmap(values,colors)
#         fg_cmap = colormap.set_alpha_except_min(new_cmap,1,False)
#         for i in np.unique(mat):
#             plt.figure(figsize=(10, 10))
#             mat_c = (mat==i).astype(np.float32)
#             if pos is not None:
#                 plt.matshow(mat_c, cmap=fg_cmap,alpha=0.7)
#                 ax=plt.scatter(pos[:,1],pos[:,0],c=pos[:,3],s=5,cmap='hot')
#                 plt.colorbar(ax)
                
#             else:
#                 plt.matshow(mat_c, cmap='jet')
#             plt.savefig(os.path.join(save_path,'kmeans_{}_{}c.png'.format(clus,i)), dpi = 400)
#             plt.cla()
#             # mat_c.astype(np.float32).tofile(os.path.join(save_path,'kmeans_{}_{}c.dat'.format(clus,i)))
def kmeans(model,train_dl,device,matshape,save_path,pos = None):
    #transformer features
    features = []
    position = []
    encoder = model[0]
    token_transformer = model[1]
    encoder.eval()
    token_transformer.eval()
    with torch.no_grad():
        for data in tqdm(train_dl):
            sample = data['sample_ori'].float().to(device)
            posi = data['position'].float().to(device)
            #sample: b*channel*seq
            x,_ = encoder(sample)
            # x = torch.max(x.squeeze(2), dim=1)[0] # use for notransformer
            x,ext_x = token_transformer(x)
            # print(sample.shape,ext_x.shape)
            # x: b * seq * dim/b * dim
            if len(x.shape)>2 and (x.shape[1]>64):
                x_flat = torch.max(x, dim=-1)[0]
            else:
                x_flat = x.reshape(x.shape[0],-1)
            features.append(x_flat)
            position.append(posi[:,-2:].cpu())
    print('output feature shape:', len(features)*len(features[0]), len(features[0][0]))
    features = torch.cat(features,dim=0)
    position = torch.cat(position,dim=0).numpy().astype(np.int32)
    print('draw feature shape:', features.shape)
    
    # pca = PCA(n_components=0.9)
    # if features.shape[1]>500:
    #     features = pca.fit_transform(features)
    # #construct tsne
    # tsne = TSNE(n_components=2)
    # k = pca.fit_transform(features[:4196])
    # print('k shape:', k.shape)
    # X_tsne = tsne.fit_transform(k)
    
    # plot similarity distribute
    # plt.cla()
    # sim = torch.nn.CosineSimilarity(dim=-1)
    # index = random.sample(range(0,features.shape[0]),4196)
    # sim_m = sim(features[index,:].unsqueeze(1), features[index,:].unsqueeze(0))
    # sns.distplot(sim_m.cpu().reshape(-1))
    # plt.xlim([-1,1])
    # plt.savefig(os.path.join(save_path,'dist.png'),dpi=400)
    # plt.cla()
    
    # plot kmeans
    mat = torch.zeros((matshape[0], matshape[1]))-1
    for clus in range(3,10):
        print('Kmeans_{}_model_{}.png'.format(clus,len(model)))
        #kmeans = MiniBatchKMeans(init='k-means++', n_clusters=clus, batch_size=10000, random_state=42, verbose=0).fit(emb_num)
        cluster_ids_x, cluster_centers = Kmeans_torch(X=features, num_clusters=clus, distance='euclidean', device=device)
        
        print(position[:,0].max(),position[:,1].max())
        mat[position[:,0],position[:,1]] = cluster_ids_x.cpu().float()

        #plot tsne
        # plt.cla()
        # plt.figure(figsize=(10, 10))
        # plt.scatter(X_tsne[:,0], X_tsne[:,1],c=kmeans.labels_[:4196])
        # plt.savefig(os.path.join(save_path,'Tsne_{}.png'.format(clus)), dpi = 400)
        # plt.cla()
        
        #plot all
        plt.figure(figsize=(10, 10))
        if pos is not None:
            ax=plt.matshow(mat,cmap='jet',alpha=0.7)
            plt.scatter(pos[:,1],pos[:,0],c=pos[:,3],s=5,cmap='hot')
            plt.colorbar(ax)
            
        else:
            plt.matshow(mat, cmap='jet')
        plt.savefig(os.path.join(save_path,'Kmeans_{}.png'.format(clus)), dpi = 400)
        plt.cla()
        mat.numpy().astype(np.float32).tofile(os.path.join(save_path,'Kmeans_{}.dat'.format(clus)))
        # #plot single
        values = [0.0,3.0]
        colors = ['blue','blue']
        new_cmap = colormap.custom_disc_cmap(values,colors)
        fg_cmap = colormap.set_alpha_except_min(new_cmap,1,False)
        for i in np.unique(mat):
            plt.figure(figsize=(10, 10))
            mat_c = (mat==i).numpy().astype(np.float32)
            if pos is not None:
                plt.matshow(mat_c, cmap=fg_cmap,alpha=0.7)
                ax=plt.scatter(pos[:,1],pos[:,0],c=pos[:,3],s=5,cmap='hot')
                plt.colorbar(ax)
                
            else:
                plt.matshow(mat_c, cmap='jet')
            plt.savefig(os.path.join(save_path,'kmeans_{}_{}c.png'.format(clus,i)), dpi = 400)
            plt.cla()
            # mat_c.astype(np.float32).tofile(os.path.join(save_path,'kmeans_{}_{}c.dat'.format(clus,i)))