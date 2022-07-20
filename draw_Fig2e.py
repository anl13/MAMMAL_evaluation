import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch 
import pickle 
from utils import * 
import os 

### eval correct body part ratio 
def compute_correct_part_ratio(est, gt, N=70, name="est"):
    part = g_all_parts
    est_data = est[0:N, :, part,:]
    gt_data  = gt[0:N, :, part, :]
    PN = len(part) 
    valid_instance = 0
    bins = np.zeros(20) 
    for fid in range(N):
        for pid in g_pig_ids_for_eval:
            correct_thresh = 0.07
            correct_num = 0 
            for k in range(PN):
                if np.linalg.norm(gt_data[fid, pid, k]) == 0: 
                    continue 
                else:
                    loss = np.linalg.norm(gt_data[fid, pid, k] - est_data[fid,pid,k])
                    if loss < correct_thresh: 
                        correct_num += 1 
            bins[correct_num] += 1
            valid_instance += 1
    for k in range(1,20):
        bins[19-k] += bins[19-k+1]
    bins = bins / valid_instance

    if not os.path.exists("data_for_fig"): 
        os.makedirs("data_for_fig")
    np.savetxt("data_for_fig/" + name +"_correctratio0.07.txt", bins)
    return bins

def load_and_eval_part_ratio(configname, N):
    with open("data/keypoints_for_eval/label_mix.pkl", 'rb') as f: 
        gt = pickle.load(f) 
    with open("data/keypoints_for_eval/" + configname + ".pkl", 'rb') as f: 
        est = pickle.load(f) 
    compute_correct_part_ratio(est, gt, N, name=configname)
    return

def eval_parts_ratio():
    config_names = [ 
        "MAMMAL", 
        "MAMMAL(5)",
        "MAMMAL(3)",
        "Tri",
        "Tri(5)", 
        "Tri(3)"
    ]

    for config in config_names: 
        load_and_eval_part_ratio(config, N=70) 

def draw_per_part_error():
    mpl.rc('font', family='Arial') 
    fig = plt.figure(figsize=(1.7,1.7)) 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

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

    folder = "data_for_fig/"
    data_MAMMAL_10 = np.loadtxt(folder + "MAMMAL_correctratio0.07.txt")
    data_MAMMAL_5 = np.loadtxt(folder +"MAMMAL(5)_correctratio0.07.txt")
    data_MAMMAL_3 = np.loadtxt(folder +"MAMMAL(3)_correctratio0.07.txt")
    data_Tri_10 = np.loadtxt(folder +"Tri_correctratio0.07.txt")
    data_Tri_5 = np.loadtxt(folder +"Tri(5)_correctratio0.07.txt")
    data_Tri_3 = np.loadtxt(folder +"Tri(3)_correctratio0.07.txt")

    labels = ["MAMMAL 10views", 
        "MAMMAL 5views", 
        "MAMMAL 3views",
        "Tri 10views",
        "Tri 5views", 
        "Tri 3views"
    ]

    colors = [ 
        colors_warm[0], colors_warm[1], colors_warm[2], 
        colors_cold[0], colors_cold[1], colors_cold[2]
    ]
    all_data = [ 
        data_MAMMAL_10, data_MAMMAL_5, data_MAMMAL_3, 
        data_Tri_10, data_Tri_5, data_Tri_3
    ]

    plt.grid(linestyle="--", alpha=0.3)
    x = np.arange(20) 
    for i in range(6):
        plt.plot(x, all_data[i], color=colors[i], marker='.', markersize=1, linewidth=0.5) 

    plt.xticks([0,5,10,15,19], fontsize=7) 
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0], labels=[0,20,40,60,80,100], fontsize=7)

    ax = fig.get_axes()[0]
    for line in ["bottom", "left", "right", "top"]: 
        ax.spines[line].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)

    plt.ylabel("Fraction of Instances (%)", fontsize=7) 
    plt.xlabel("Number of Correct Keypoints", fontsize=7)
    
    plt.text(0.02, 0.06, "Threshold: 7 cm\n(10% body length)", fontsize=7)
    plt.savefig("figs/Fig.2e.png", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    # plt.savefig("figs/Fig.2e.svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)

if __name__ == "__main__":
    if not os.path.exists("figs"):
        os.makedirs("figs")
    eval_parts_ratio()
    draw_per_part_error() 