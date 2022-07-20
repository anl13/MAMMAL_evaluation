import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from utils import *
from scipy.stats import sem 
import pickle 
import os 

def compute_loss(est, gt, N=70, name="est", part=g_all_parts, part_names="all_parts"):
    est_data = est[0:N, :, part,:]
    gt_data  = gt[0:N, :, part, :]
    PN = len(part) 
    valid_num = 0 
    error_per_point = [] 
    errors = np.zeros((N,4,PN))
    for fid in range(N):
        for pid in g_pig_ids_for_eval:
            for k in range(PN):
                if np.linalg.norm(gt_data[fid, pid, k]) == 0: 
                    errors[fid, pid, k] = -1
                    continue 
                else: 
                    valid_num += 1 
                loss = np.linalg.norm(gt_data[fid, pid,k] - est_data[fid,pid,k])
                error_per_point.append(loss) 
                errors[fid, pid, k] = loss
                
    error_per_point = np.asarray(error_per_point) 
    outfilename = "data_for_fig/" + name + "_" + part_names + ".txt" 
    np.savetxt(outfilename, error_per_point)
    
    avg = error_per_point.mean() 
    print("avg loss        : ", avg, "m") 
    print("std             : ", error_per_point.std())

def compute_PCK_curve(est, gt, N=70, name="est"):
    part = g_all_parts
    est_data = est[0:N, :, part,:]
    gt_data  = gt[0:N, :, part, :]
    PN = len(part) 
    valid_num = 0 
    thresh = np.linspace(0.0, 0.08, 17)
    bins = np.zeros(thresh.shape[0]) 
    for fid in range(N):
        for pid in g_pig_ids_for_eval:
            for k in range(PN):
                partid = part[k]
                if np.linalg.norm(gt_data[fid, pid, k]) ==0: 
                    continue 
                else: 
                    valid_num += 1 
                if np.linalg.norm(est_data[fid, pid,k]) ==0: 
                    continue 
                else:
                    loss = np.linalg.norm(gt_data[fid, pid,k] - est_data[fid,pid,k])
                    for l in range(thresh.shape[0]): 
                        if loss < thresh[l]: 
                            bins[l] += 1
                            break 
                    if loss > thresh[-1]: 
                        # frameid = 750 + fid * 25
                        # print("frame ", frameid, " pig ", pid, g_jointnames[partid], " error: ", loss)
                        # print("gt : ", gt_data[fid, pid, k])
                        # for tmp_pid in range(4):
                        #     print("est: ", est_data[fid, tmp_pid, k], "  for pig ", tmp_pid)    
                        pass 
    print("area under curve: ", bins.sum() / valid_num)
    for m in range(thresh.shape[0] - 1):
        bins[m+1] = bins[m+1] + bins[m]
    bins = bins / valid_num 
    
    return bins 
    

def load_and_eval(configname, N):
    with open("data/keypoints_for_eval/label_mix.pkl", 'rb') as f: 
        gt = pickle.load(f) 
    with open("data/keypoints_for_eval/" + configname + ".pkl", 'rb') as f: 
        est = pickle.load(f) 
    compute_loss(est, gt, N=N, name=configname, part=g_all_parts, part_names="all_parts_mix") 
    return compute_PCK_curve(est, gt, N=N, name=configname)

def eval_compare():
    config_names = [ 
        "MAMMAL", 
        "MAMMAL(5)",
        "MAMMAL(3)",
        "Tri",
        "Tri(5)", 
        "Tri(3)"
    ]
    mpl.rc('font', family='Arial') 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    f, ax = plt.subplots(nrows=1, ncols=1)
    f.set_figheight(1.7)
    f.set_figwidth(1.7)
    plt.grid(linestyle="--", alpha=0.3)
    colors_warm = np.asarray([
        [227, 88, 34],
        [227, 119, 34],
        [227, 160, 34]    
    ]) / 255.0 
    colors_cold = np.asarray([
        [34, 75, 227], 
        [34, 121, 227], 
        [34, 185, 227]
    ]) / 255.0
    colors = [ 
        colors_warm[0], colors_warm[1], colors_warm[2], 
        colors_cold[0], colors_cold[1], colors_cold[2]
    ]
    xs = np.linspace(0.0, 0.08, 17)
    labels = []
    all_data = [] 
    for name in config_names: 
        labels.append(name)
        print("......eval folder: ", name, "....")
        bins = load_and_eval(name, N=70)
        all_data.append(bins)
    for i in range(6):
        ax.plot(xs, all_data[i], color=colors[i], marker=".", markersize=1, linewidth=0.5)
    for line in ["bottom", "left", "right", "top"]: 
        ax.spines[line].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    ax.set_ylim(-0.05,1.05)
    plt.xticks([0,0.02,0.04,0.06,0.08], labels=[0,2,4,6,8], fontsize=7) 
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0], labels=[0,20,40,60,80,100], fontsize=7)
    plt.xlabel("Error Threshold (cm)", fontsize=7)
    plt.ylabel("Accuracy (%)", fontsize=7)
    plt.savefig("figs/Fig.2f.png", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    # plt.savefig("figs/Fig.2f.svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)

if __name__ == "__main__":
    if not os.path.exists("figs"):
        os.makedirs("figs")
    eval_compare() 
