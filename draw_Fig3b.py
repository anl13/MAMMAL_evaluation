import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import os 
from utils import *
import seaborn as sns 
import pandas as pd 
from matplotlib.patches import Patch 
from matplotlib.lines import Line2D
import pickle 

# dist : 
# leye to reye: 0.071
# lear to rear: 0.078
# nose to leye: 0.096
# leblow to lpaw: 0.065
# nose to lear: 0.168
### compute body part errors 
def compute_per_part_loss(est, gt, N=70, name="est", part=g_all_parts, part_names="all_parts", gt_vis=None):
    est_data = est[0:N, :, part,:]
    gt_data  = gt[0:N, :, part, :]
    gt_vis_data = gt_vis[0:N, :, part, :]
    PN = len(part) 
    errors = np.zeros((N, 4, PN))

    # eye_dist = [] 
    for fid in range(N):
        for pid in g_pig_ids_for_eval:
            # dist = np.linalg.norm(gt_data[fid, pid, 1] - gt_data[fid, pid, 2])
            # eye_dist.append(dist)
            for k in range(PN):
                if np.linalg.norm(gt_vis_data[fid, pid, k]) == 0: 
                    loss = np.linalg.norm(gt_data[fid, pid,k] - est_data[fid,pid,k])
                    errors[fid, pid,k] = - loss
                else: 
                    loss = np.linalg.norm(gt_data[fid, pid,k] - est_data[fid,pid,k])
                    errors[fid, pid, k] = loss
    # eye_dist = np.asarray(eye_dist) 
    # print("eye dist mean:", eye_dist.mean())
    if not os.path.exists("data_for_fig"): 
        os.makedirs("data_for_fig") 
    outfilename = "data_for_fig/" + name + "_" + part_names + ".txt" 
    out = errors.copy() 
    out = out.reshape([N*4, -1])
    print(out.shape)
    np.savetxt(outfilename, out)

    for k in range(PN):
        jid = part[k]
        part_e = errors[:,:,k]
        avg = part_e[part_e >= 0].mean() # avg for visible ones 
        avg_total = np.abs(part_e).mean() 
    #     print(g_jointnames[jid],":", avg_total, " (m)")
    # print("all:", out[out>=0].mean())


def load_and_eval(configname, N):
    with open("data/keypoints_for_eval/label_mix.pkl", 'rb') as f: 
        gt = pickle.load(f) 
    with open("data/keypoints_for_eval/label_3d.pkl", 'rb') as f: 
        gt_vis = pickle.load(f) 
    with open("data/keypoints_for_eval/" + configname + ".pkl", 'rb') as f: 
        est = pickle.load(f) 
    compute_per_part_loss(est, gt, N, name=configname, part=g_all_parts, part_names="all_mix", gt_vis=gt_vis)
    return

def eval():
    config_names = [ 
        "MAMMAL", 
        "MAMMAL(5)",
        "MAMMAL(3)",
        "Tri",
        "Tri(5)", 
        "Tri(3)"
    ]

    for config in config_names: 
        load_and_eval(config, N=70) 

def build_dataframe():
    folder = "data_for_fig/"
    data_MAMMAL_10 = np.loadtxt(folder + "MAMMAL_all_mix.txt")
    d1 = data_MAMMAL_10
    errors = []
    method = [] 
    jointname = [] 
    part = g_all_parts 
    groupname = [] 
    visible = [] 
    for i in range(len(part)):
        id = g_all_parts[i]
        name = g_jointnames[id] 
        gn = g_groupnames[id]
        for k in range(d1.shape[0]): 
            if d1[k,i] >= 0: 
                visible.append(0)
                errors.append(d1[k,i])
            else: 
                visible.append(1)
                errors.append(-d1[k,i]) 
            jointname.append(name)
            method.append("MAMMAL") 
            groupname.append(gn)
    d = {"error": errors, "method": method, "jointname": jointname, "groupname": groupname, "visible":visible}
    return pd.DataFrame(data=d)

def draw(): 
    mpl.rc('font', family='Arial') 

    fig = plt.figure(figsize=(3.3,1.7)) 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    colors = {
        0: np.asarray([1,1,1]),
        1: np.asarray([0.7,0.7,0.7])
    }

    data = build_dataframe()
    useddata = data
    # ax = sns.violinplot(x="jointname", y="error", hue="visible", data=useddata, palette="Set3",
    #     split=False, inner="quartile", linewidth=1)

    ax = sns.boxplot(x="jointname", y="error", hue="visible", data=useddata, palette=colors,
        linewidth=0.5, sym="", showmeans=True,
        
        meanprops = {'marker':'s','markerfacecolor':'black','markeredgecolor':'black', 'linewidth':0, 'markersize':1.5}, 
        capprops={"linewidth":0.5, "color": 'k'},
        whiskerprops={"linewidth":0.5, "color": "k"},
        )
    ax.plot([-0.67,18.23], [0.07,0.07], linestyle='--',linewidth=0.5, color = 'g')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='black', label="Visible Keypoints", linewidth=0.5), 
        Patch(facecolor=colors[1], edgecolor='black',label="Invisible Keypoints", linewidth=0.5),
        Line2D([0], [0], color=(0,0.5,0), linestyle='--', lw=0.5, label="10% body length")
    ]
    ax = fig.get_axes()[0]
    for line in ["bottom", "left", "right", "top"]: 
        ax.spines[line].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)

    plt.legend(handles=legend_elements, fontsize=6, ncol=1, loc='upper left', bbox_to_anchor=(0.06, 1.01), frameon=False)
    plt.xlabel("", fontsize=7)
    plt.ylabel("Error (cm)", fontsize=7)
    plt.ylim(0,0.25)
    plt.yticks([0,0.05,0.1,0.15,0.2,0.25], labels=[0,5,10,15,20,25], fontsize=7)
    plt.savefig("figs/Fig.3b.png", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    # plt.savefig("figs/Fig.3b.svg", dpi=1000, bbox_inches='tight', pad_inches=0.01) # uncomment this to write vector image.

if __name__ == "__main__":
    if not os.path.exists("figs"):
        os.makedirs("figs")
    eval()
    draw()
