import os
import numpy as np
from utils import *
from tqdm import tqdm
def shengli_data(name,window,step,norm,shift=0,ep=2,property='sx',train_mode='ssl',train_dataset='train',ext=1,t=1):
    n3, n2, n1 = 511,621,601
    path = 'data/shengli/'
    if train_mode == 'ssl' or train_mode == 'predict':
        if os.path.exists(path+name+'s_'+property+"_ssldata_window_"+str(window)+"shift_"+str(shift)+"step_"+str(step)+".dat"):
            print("Load dat : "+property+"_ssldata_window_"+str(window)+"shift_"+str(shift)+"step_"+str(step)+".dat")
            train_dat = np.fromfile(path+name+'_'+property+"_ssldata_window_"+str(window)+"shift_"+str(shift)+"step_"+str(step)+".dat", np.float32).reshape(-1,2*window+1)
            train_pos = np.fromfile(path+name+'_'+property+"_ssldata_window_"+str(window)+"shift_"+str(shift)+"step_"+str(step)+"_pos.dat", np.float32).reshape(-1,3)
        else:
            print("Make Data From Raw Data")
            sxp = path + property +'.dat'
            sx = np.fromfile(sxp, dtype = np.float32).reshape(n3, n2, n1)
            if norm:
                sx = xnorm(sx)
            if name == 'shengli_ngs3':
                hor = np.fromfile(path+'ngs3.dat',dtype=np.float32).reshape(n3,n2)
                hor[hor<0]=0
            elif name == 'shengli_ngs4':
                hor = np.fromfile(path+'ngs4.dat',dtype=np.float32).reshape(n3,n2)
                hor[hor<0]=0
            train_dat = []
            train_pos = []
            inline = np.arange(0,n3)
            inline_zs = (inline - np.mean(inline))/np.std(inline)
            xline = np.arange(0,n2)
            xline_zs = (xline - np.mean(xline))/np.std(xline)
            
            ext = ext * t
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
                                    h = round(hor[ix,xx]) + shift
                                    s.append(sx[ix][xx][h-window:h+window+1].tolist())#+[inline_zs[ix],xline_zs[xx]]
                            l.append(s)
                            p.append([inline_zs[i3],xline_zs[i2],round(hor[i3,i2]),i3,i2])
                    l = np.array(l,dtype=np.float32)
                    p = np.array(p,dtype=np.float32)
                    train_dat.append(l)
                    train_pos.append(p)
            train_dat = np.concatenate(train_dat,axis=0).astype(np.float32)
            train_pos = np.concatenate(train_pos,axis=0).astype(np.float32)
            # train_dat.tofile(path+name+'_'+property+"_ssldata_window_"+str(window)+"shift_"+str(shift)+"step_"+str(step)+".dat")
            # train_pos.tofile(path+name+'_'+property+"_ssldata_window_"+str(window)+"shift_"+str(shift)+"step_"+str(step)+"_pos.dat")
            point1 = [[260,207],[190,486],[426,253],[149,521]]
            point2 = [[387,285],[224,339],[239,407],[51,481],[164,397]]
            point3 = [[149,442],[101,421]]
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
                        h = round(hor[ix,xx]) + shift
                        s.append(sx[ix][xx][h-window:h+window+1].tolist())#+[inline_zs[ix],xline_zs[xx]]
                train_label_data.append(s)
                train_label_label.append([inline,xline,labels[n]])
    else:
        if os.path.exists(path+name+'s_'+property+"_ftdata_window_"+str(window)+"shift_"+str(shift)+"ep_"+str(ep)+".dat"):
            print("Load dat : "+property+"_ftdata_window_"+str(window)+"shift_"+str(shift)+"step_"+str(step)+".dat")
            train_dat = np.fromfile(path+name+'_'+property+"_ftdata_window_"+str(window)+"shift_"+str(shift)+"ep_"+str(ep)+".dat", np.float32).reshape(-1,2*window+1)
            train_pos = np.fromfile(path+name+'_'+property+"_ftdata_window_"+str(window)+"shift_"+str(shift)+"ep_"+str(ep)+"_pos.dat", np.float32).reshape(-1,4)
        else:
            hlabel = np.fromfile(path+'labels_'+name.split('_')[-1]+'.dat',dtype=np.float32).reshape(n3,n2)
            
            sxp = path + property+'.dat'
            sx = np.fromfile(sxp, dtype = np.float32).reshape(n3, n2, n1)
            if norm:
                sx = xnorm(sx)
            if name == 'shengli_ngs3':
                hor = np.fromfile(path+'ngs3.dat',dtype=np.float32).reshape(n3,n2)
                hor[hor<0]=0
                label_dict = {1:0,2:1,3:2}
            wp = []
            for inline in range(n3):
                for xline in range(n2):
                    if hlabel[inline][xline] > 0:
                        # print(inline,xline,down[inline,xline],labeld[inline][xline])
                        wp.append([inline,xline,hor[inline,xline],hlabel[inline][xline]])
            wp = np.array(wp)
            wlabel = list(set(wp[:,3]))
            dat = []
            dlabel = []
            dpos = []
            inline = np.arange(0,n3)
            inline_zs = (inline - np.mean(inline))/np.std(inline)
            xline = np.arange(0,n2)
            xline_zs = (xline - np.mean(xline))/np.std(xline)
            for i,info in enumerate(wp):
                inline = int(info[0])
                xline = int(info[1])
                dep = round(info[2])
                l = label_dict[info[3]]
                if dep>0:
                    for j1 in range(-ep, ep+1):
                        for j2 in range(-ep, ep+1):
                            d = sx[inline+j1][xline+j2][dep-window+shift:dep+window+1+shift].tolist()
                            dat.append(d+[inline_zs[inline],xline_zs[inline]])
                            dlabel.append(l)
                            dpos.append([inline,xline,dep,l])
            train_dat = np.array(dat).astype(np.float32)
            train_label = np.array(dlabel).astype(np.float32)
            train_pos = np.array(dpos).astype(np.float32)
            train_label_data = []
            train_label_label = []
            # train_dat.tofile(path+name+'_'+property+"_ftdata_window_"+str(window)+"shift_"+str(shift)+"ep_"+str(ep)+".dat")
            # train_pos.tofile(path+name+'_'+property+"_ftdata_window_"+str(window)+"shift_"+str(shift)+"ep_"+str(ep)+"_pos.dat")  
    return train_dat ,train_pos,np.array(train_label_data),np.array(train_label_label)
