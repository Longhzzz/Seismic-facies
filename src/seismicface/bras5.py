import os
import numpy as np
from utils import *
from tqdm import tqdm
def jintao_data(name,window,step,norm,property='sx',ep=2,train_mode='ssl',train_dataset='train'):
    n3,n2,n1 = 640,1280,1001
    path = 'data/bras/canal0501/'
    if os.path.exists(path+"_"+property+"_ssldata_window_"+str(window)+"step_"+str(step)+".dat"):
        print("Load dat : "+"ssldata_window_"+str(window)+"step_"+str(step)+".dat")
        train_dat = np.fromfile(path+"_"+property+"ssldata_window_"+str(window)+"step_"+str(step)+".dat", np.float32).reshape(-1,2*window+1+2)
        train_pos = np.fromfile(path+"_"+property+"ssldata_window_"+str(window)+"step_"+str(step)+"_pos.dat", np.float32).reshape(-1,3)
    else:
        print("Make Data From Raw Data")
        sxp = path + property+".dat"
        sx = np.fromfile(sxp, dtype = np.float32).reshape(n3, n2, n1)
        if norm:
            sx = xnorm(sx)
        horp = path + "hor.dat"
        hor = np.fromfile(horp, dtype = np.float32).reshape(n3,n2)
        train_dat = []
        train_pos = []
        inline = np.arange(0,n3)
        inline_zs = (inline - np.mean(inline))/np.std(inline)
        xline = np.arange(0,n2)
        xline_zs = (xline - np.mean(xline))/np.std(xline)
        ext = 6
        t = int(ext/3)
        if ext == 0:
            t = 3
        # if train_mode == 'predict':
        #     ext = 0
        for i3 in tqdm(range(0,n3,step)):
            if i3 - ext > 0 and i3 + ext < n3:
                l = []
                p = []
                for i2 in range(0,n2,step):
                    if i2 - ext > 0 and i2 + ext < n2:
                        s = []
                        for i21 in range(-ext,ext+1,t):
                            for i22 in range(-ext,ext+1,t):
                                ix = i3 + i21
                                xx = i2 + i22
                                h = round(hor[ix,xx])
                                s.append(sx[ix][xx][h-window:h+window+1].tolist())#+[inline_zs[ix],xline_zs[xx]]
                        l.append(s)
                        p.append([i3,i2,round(hor[i3,i2])])
                l = np.array(l,dtype=np.float32)
                p = np.array(p,dtype=np.float32)
                train_dat.append(l)
                train_pos.append(p)
        train_dat = np.concatenate(train_dat,axis=0).astype(np.float32)
        train_pos = np.concatenate(train_pos,axis=0).astype(np.float32)
        # train_dat.tofile(path+"_"+property+"_ssldata_window_"+str(window)+"step_"+str(step)+".dat")
        # train_pos.tofile(path+"_"+property+"_ssldata_window_"+str(window)+"step_"+str(step)+"_pos.dat")
    
    point1 = [[60,410],[80,410],[100,400],[120,390],[140,380],[160,380],[200,400],[230,450],[250,500],[270,520],
         [290,570],[310,610],[330,660],[350,690],[370,710],[390,720],[100,900],[150,920],[200,940],[250,930]]
    point2 = [[50,350],[70,330],[120,440],[140,460],[160,480],[180,500],[200,530],[220,560],[240,580],
            [320,540],[340,510],[360,490],[380,480],[400,470],[420,450],[380,550],[400,570],[420,570],
            [470,570],[490,580]]
    point3 = [[100,1350],[150,1400],[200,1430],[250,1460],[250,1500],[260,1570],[280,1600],[330,1650],
            [380,1700],[290,450],[270,430],[250,410],[270,390],[290,410],[310,430],[290,370],[310,390],
            [330,410],[300,350],[300,330]]
    points = np.array(point1+point2+point3)
    labels = np.concatenate([np.ones(len(point1))*0,np.ones(len(point2))*1,np.ones(len(point3))*2],axis=0)
    train_label_data = []
    train_label_label = []
    for n,i in enumerate(points):
        inline = i[0]
        xline = i[1]
        s = []
        for i21 in range(-ext,ext+1,t):
            for i22 in range(-ext,ext+1,t):
                ix = inline + i21
                xx = xline + i22
                h = round(hor[ix,xx])
                s.append(sx[ix][xx][h-window:h+window+1].tolist())#+[inline_zs[ix],xline_zs[xx]]
        train_label_data.append(s)
        train_label_label.append([inline,xline,labels[n]])
    return train_dat ,train_pos, np.array(train_label_data),np.array(train_label_label)