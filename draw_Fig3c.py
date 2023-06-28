import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import os 
from utils import *
import seaborn as sns 
import pandas as pd 
from matplotlib.patches import Patch 
import matplotlib.cm as cm 
from scipy.stats import sem 

def load_gt_area(): 
    folder = "data/iou_for_eval/GT/"
    area = [] 
    for k in range(70): 
        frameid = 750 + 25 * k 
        filename = folder + "Area_{}.txt".format(frameid) 
        gt = np.loadtxt(filename) 
        area.append(gt) 
    area = np.asarray(area) 
    return area 

g_areas = load_gt_area() 

def load_sil_loss(folder, startid):
    ious = []
    for k in range(70):
        frameid = startid + 25 * k 
        ifile = folder + "I_{}.txt".format(frameid)
        Is = np.loadtxt(ifile) 
        ufile = folder + "U_{}.txt".format(frameid)
        Us = np.loadtxt(ufile) 
        Us_nozero = Us.copy() 
        Us_nozero[Us == 0] = 1 
        iou = Is / Us_nozero 
        iou[Us==0] = -1 
        ious.append(iou)

    ious = np.asarray(ious) 
    return ious 

def build_data_frame_chronic():
    folders = [
        ["data/iou_for_eval/MAMMAL/", "MAMMAL",0],
        ["data/iou_for_eval/MAMMAL_no_sil_new/", "-sil",0],
    ]
    N = len(folders)
    out_ious = []
    out_method_label = [] 
    out_frame_label = [] 
    
    for k in range(N):
        print(folders[k][1])
        ious = load_sil_loss(folders[k][0], folders[k][2]) 
        for fid in range(70): 
            for pid in range(4):
                for camid in range(10):
                    if ious[fid, pid, camid] < 0: 
                        continue 
                    if g_areas[fid, pid, camid] == 0: 
                        continue 
                    out_ious.append(ious[fid, pid, camid])
                    out_method_label.append(folders[k][1])
                    out_frame_label.append(fid)
    
    data_dict = { 
        "iou": out_ious, 
        "method": out_method_label,
        "time": out_frame_label
    }
    return pd.DataFrame(data=data_dict) 

def draw_sil_curve(): 
    mpl.rc('font', family='Arial') 
    fig = plt.figure(figsize=(1.6,1.4)) 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    ax = fig.add_subplot(111)
    data = build_data_frame_chronic()

    names = ["MAMMAL", "-sil"]
    legend_names = ["MAMMAL", "MAMMAL w/o Sil"]
    data_per_frame = [] 
    sem_per_frame  = [] 

    for i in range(70):
        d = np.zeros(2)
        s = np.zeros(2)
        for k in range(2):
            raw = data[data.time==i]
            method = raw[raw.method==names[k]]
            method = method.iou.values
            mean_data = method.mean()
            sem_data = sem(method) 
            # sem_data = np.std(method) # standard deviation 
            d[k] = mean_data 
            s[k] = sem_data 
        data_per_frame.append(d)
        sem_per_frame.append(s)  
    data_per_frame = np.asarray(data_per_frame)
    sem_per_frame  = np.asarray(sem_per_frame) 
    
    colortable = np.asarray([ 
        [254, 138, 113],
        [173, 203, 227]
    ])/ 255 
    ax = fig.get_axes()[0]
    xs = np.arange(70) 
    for index, k in enumerate([0,1]): 
        ax.plot(xs, data_per_frame[:,k],  label=legend_names[index], color=colortable[k],linewidth=0.5)
        up = data_per_frame[:,k] + sem_per_frame[:,k]
        down = data_per_frame[:,k] - sem_per_frame[:,k]
        ax.fill_between(xs, down, up, color=colortable[k], alpha=0.25, linewidth=0)
    plt.grid(linestyle='--', alpha=0.3)
    plt.legend(fontsize=7, frameon=False, ncol=1, loc=3)
    plt.xticks([0,10,20,30,40,50,60,70], fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylim(0.6,0.87)
    for line in ["bottom", "left", "right", "top"]: 
        ax.spines[line].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)

    plt.xlabel("Time (s)", fontsize=7) 
    plt.ylabel("Surface Accuracy (IoU)", fontsize=7)

    plt.savefig("figs/Fig.3c.png", dpi=1000, pad_inches=0.01, bbox_inches='tight')
    plt.savefig("figs/Fig.3c.svg", dpi=1000, pad_inches=0.01, bbox_inches='tight')


if __name__ == "__main__":
    if not os.path.exists("figs"):
        os.makedirs("figs")
    draw_sil_curve() 
