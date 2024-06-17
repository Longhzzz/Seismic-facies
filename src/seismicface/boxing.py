import os
import numpy as np
from utils import *
from tqdm import tqdm
def boxing_data(name,window,step,norm,shift=0,ep=2,property='sx',train_mode='ssl',train_dataset='train'):
    n3, n2, n1 = 1121,881,1251
    path = 'data/boxing/'
    if train_mode == 'ssl' or train_mode == 'predict':
        if os.path.exists(path+name+'_'+property+"s_ssldata_window_"+str(window)+"shift_"+str(shift)+"step_"+str(step)+".dat"):
            print("Load dat : "+property+"_ssldata_window_"+str(window)+"step_"+str(step)+".dat")
            train_dat = np.fromfile(path+name+'_'+property+"_ssldata_window_"+str(window)+"shift_"+str(shift)+"step_"+str(step)+".dat", np.float32).reshape(-1,2*window+1)
            train_pos = np.fromfile(path+name+'_'+property+"_ssldata_window_"+str(window)+"shift_"+str(shift)+"step_"+str(step)+"_pos.dat", np.float32).reshape(-1,3)
        else:
            print("Make Data From Raw Data")
            sxp = path + property +'.dat'
            sx = np.fromfile(sxp, dtype = np.float32).reshape(n3, n2, n1)
            if norm:
                sx = xnorm(sx)
            if name == 'boxing_es4cs2':
                hor = (np.fromfile(path+'New_T6xb1.dat',dtype=np.float32).reshape(n3,n2)-1000)/2
                hor[hor<0]=0
            elif name == 'boxing_es4cs1':
                hor = (np.fromfile(path+'New_T6x.dat',dtype=np.float32).reshape(n3,n2)-1000)/2
                hor[hor<0]=0
            else:
                hor = (np.fromfile(path+'T7_new_5.dat',dtype=np.float32).reshape(n3,n2)-1000)/2
                hor[hor<0]=0
            train_dat = []
            train_pos = []
            inline = np.arange(0,n3)
            inline_zs = (inline - np.mean(inline))/np.std(inline)
            xline = np.arange(0,n2)
            xline_zs = (xline - np.mean(xline))/np.std(xline)
            for i3 in tqdm(range(0,n3,step)):
                for i2 in range(0,n2,step):
                    h = round(hor[i3,i2])
                    if h>1:
                        train_dat.append(sx[i3][i2][h-window+shift:h+window+1+shift].tolist()+[inline_zs[i3],xline_zs[i2]])
                        train_pos.append([i3,i2,h])
            train_dat = np.array(train_dat).astype(np.float32)
            train_pos = np.array(train_pos).astype(np.float32)
            train_dat.tofile(path+name+'_'+property+"_ssldata_window_"+str(window)+"shift_"+str(shift)+"step_"+str(step)+".dat")
            train_pos.tofile(path+name+'_'+property+"_ssldata_window_"+str(window)+"shift_"+str(shift)+"step_"+str(step)+"_pos.dat")
    else:
        if os.path.exists(path+name+'s_'+property+"s_ftdata_window_"+str(window)+"shift_"+str(shift)+"ep_"+str(ep)+".dat"):
            print("Load dat : "+property+"_ftdata_window_"+str(window)+"step_"+str(step)+".dat")
            train_dat = np.fromfile(path+name+'_'+property+"_ftdata_window_"+str(window)+"shift_"+str(shift)+"ep_"+str(ep)+".dat", np.float32).reshape(-1,2*window+1)
            train_pos = np.fromfile(path+name+'_'+property+"_ftdata_window_"+str(window)+"shift_"+str(shift)+"ep_"+str(ep)+"_pos.dat", np.float32).reshape(-1,4)
        else:
            lithHor = np.load(path+'lithHor.npz')
            hlabel = lithHor[name.split('_')[-1]]
            sxp = path + property+'.dat'
            sx = np.fromfile(sxp, dtype = np.float32).reshape(n3, n2, n1)
            if norm:
                sx = xnorm(sx)
            if name == 'boxing_es4cs2':
                hor = (np.fromfile(path+'New_T6xb1.dat',dtype=np.float32).reshape(n3,n2)-1000)/2
                hor[hor<0]=0
                label_dict = {21:0,
                        24:1,
                        26:2}
            elif name == 'boxing_es4cs1':
                hor = (np.fromfile(path+'New_T6x.dat',dtype=np.float32).reshape(n3,n2)-1000)/2
                hor[hor<0]=0
                label_dict = {21:0,
                        24:1,
                        26:2}
            else:
                hor = (np.fromfile(path+'T7_new_5.dat',dtype=np.float32).reshape(n3,n2)-1000)/2
                hor[hor<0]=0
                label_dict = {21:0,
                        24:1,
                        25:2,
                        26:3}
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
                            d = sx[inline+j1][xline+j2][dep-window+shift:dep+window+1+shift].tolist()+[inline_zs[inline],xline_zs[xline]]
                            dat.append(d)
                            dlabel.append(l)
                            dpos.append([inline,xline,dep,l])
            train_dat = np.array(dat).astype(np.float32)
            train_label = np.array(dlabel).astype(np.float32)
            train_pos = np.array(dpos).astype(np.float32)
            # train_dat.tofile(path+name+'_'+property+"_ftdata_window_"+str(window)+"shift_"+str(shift)+"ep_"+str(ep)+".dat")
            # train_pos.tofile(path+name+'_'+property+"_ftdata_window_"+str(window)+"shift_"+str(shift)+"ep_"+str(ep)+"_pos.dat")  
    return train_dat ,train_pos
