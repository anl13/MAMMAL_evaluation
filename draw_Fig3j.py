import matplotlib.pyplot as plt 
import numpy as np 
import pickle 
import os
from scipy.stats import sem 
from matplotlib import cm 
import matplotlib as mpl 
from utils import g_all_parts, g_jointnames

def fast_eval(MAMMAL, Tri, GT, pid = 0, jointid = 0): 
    R = np.arange(0,1750,25)
    MAMMAL = MAMMAL[R,pid,jointid]
    Tri = Tri[R,pid,jointid]
    GT = GT[:,pid,jointid]

    error_M = np.linalg.norm(MAMMAL - GT, axis=1)
    error_T = np.linalg.norm(Tri - GT, axis=1)
    print("M: ", error_M.mean(), error_M.std()) 
    print("T: ", error_T.mean(), error_T.std()) 


with open("data/traj/MAMMAL.pkl", 'rb') as f: 
    MAMMAL = pickle.load(f)  # 1750, 4, 23, 3
with open("data/traj/Tri.pkl", 'rb') as f: 
    Tri = pickle.load(f)  # 1750, 4, 23, 3
with open("data/keypoints_for_eval/label_mesh.pkl", 'rb') as f: 
    Gt = pickle.load(f)  # 70, 4, 23,3

def draw_xy_tail():
    jointid = 18
    pid = 0

    # for k in range(23): 
    #     if k not in g_all_parts: 
    #         continue 
    #     print("joint id : ", k, " ", g_jointnames[k])
    #     fast_eval(MAMMAL, Tri, Gt, pid=pid, jointid=k)

    R = np.arange(1100,1500)
    R_gt = np.arange(44,60)
    coords_M_raw = MAMMAL[R,pid,jointid]
    coords_T_raw = Tri[R,pid,jointid]
    coords_G = Gt[R_gt,pid,jointid]

    L = R.shape[0]
    blues = cm.get_cmap(plt.get_cmap('Blues'))(np.linspace(0.0,1.0,L))[:,:3]
    reds = cm.get_cmap(plt.get_cmap('Reds'))(np.linspace(0.0,1.0,L))[:,:3]

    mpl.rc('font', family='Arial') 
    fig = plt.figure(figsize=(1.4,0.9)) 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.plot(coords_M_raw[:,0], coords_M_raw[:,1],color="blue", marker='', linewidth=0.1)
    plt.scatter(coords_M_raw[:,0], coords_M_raw[:,1], s=0.15, c=blues, label='MAMMAL')
    x = np.ma.masked_where(coords_T_raw[:,0] == 0,coords_T_raw[:,0])
    y = np.ma.masked_where(coords_T_raw[:,1] == 0,coords_T_raw[:,1])

    plt.plot(x, y,color='red', marker='',linewidth=0.1)
    plt.scatter(x,y,s=0.15, c=reds,  label='Tri')
    plt.scatter(coords_G[:,0], coords_G[:,1], zorder=5, color='black', marker='^', s=1.2, label="Ground Truth")
    plt.xlim(-0.6,1.1)
    plt.ylim(-1,-0.1)
    plt.xticks([0,1], ["0","1"], fontsize=7)
    plt.yticks([-0.5,-1], ["-0.5", "-1"], fontsize=7)
    plt.xlabel("x (m)", fontsize=7) 
    plt.ylabel("y (m)", fontsize=7)
    ax = fig.get_axes()[0] 
    for line in ["bottom", "left", "right", "top"]: 
        ax.spines[line].set_linewidth(0.5)
    for line in ["top","right"]:
        ax.spines[line].set_visible(False)

    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    ax.plot(1.0, -1, ">k", markersize=0.8, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(-0.6,1, "^k", markersize=0.8, transform=ax.get_xaxis_transform(), clip_on=False)

    plt.savefig("figs/Supp.9.b.xy.png", dpi=1000, pad_inches=0.01, bbox_inches='tight')
    # plt.savefig("figs/Supp.9.b.xy.svg", dpi=1000, pad_inches=0.01, bbox_inches='tight')

def draw_z_tail():
    jointid = 18
    pid = 0
    R = np.arange(1100,1500)
    R_gt = np.arange(44,60)
    coords_M_raw = MAMMAL[R,pid,jointid]
    coords_T_raw = Tri[R,pid,jointid]
    coords_G = Gt[R_gt,pid,jointid]

    L = R.shape[0]
    blues = cm.get_cmap(plt.get_cmap('Blues'))(np.linspace(0.0,1.0,L))[:,:3]
    reds = cm.get_cmap(plt.get_cmap('Reds'))(np.linspace(0.0,1.0,L))[:,:3]

    mpl.rc('font', family='Arial') 
    fig = plt.figure(figsize=(1.4,0.9)) 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.plot(R/25, coords_M_raw[:,2], color='blue', linewidth=0.1, markersize=0)
    plt.scatter(R/25, coords_M_raw[:,2], color=blues, s=0.18)
    z = np.ma.masked_where(coords_T_raw[:,2] == 0, coords_T_raw[:,2])
    plt.plot(R/25, z, color='red', linewidth=0.1, markersize=0)
    plt.scatter(R/25, z, color=reds, s=0.18)
    plt.scatter(R_gt, coords_G[:,2], color='black', s=1.2, marker='^', zorder=5)

    # plt.show()
    plt.xlim(44,60)
    plt.ylim(0.2,0.4)
    plt.xticks([44, 52, 60], fontsize=7)
    plt.yticks([0.2,0.3,0.4], fontsize=7)
    plt.xlabel("Time (s)", fontsize=7)
    plt.ylabel("z (m)", fontsize=7)
    ax = fig.get_axes()[0] 
    for line in ["bottom", "left", "right", "top"]: 
        ax.spines[line].set_linewidth(0.5)
    for line in ["top","right"]:
        ax.spines[line].set_visible(False)

    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)

    plt.savefig("figs/Supp.9.b.z.png", dpi=1000, pad_inches=0.01, bbox_inches='tight')
    # plt.savefig("figs/Supp.9.b.z.svg", dpi=1000, pad_inches=0.01, bbox_inches='tight')


def draw_xy_shoulder():
    jointid = 6
    pid = 0
    R = np.arange(1100,1500)
    R_gt = np.arange(44,60)
    coords_M_raw = MAMMAL[R,pid,jointid]
    coords_T_raw = Tri[R,pid,jointid]
    coords_G = Gt[R_gt,pid,jointid]
    L = R.shape[0]
    blues = cm.get_cmap(plt.get_cmap('Blues'))(np.linspace(0.0,1.0,L))[:,:3]
    reds = cm.get_cmap(plt.get_cmap('Reds'))(np.linspace(0.0,1.0,L))[:,:3]

    mpl.rc('font', family='Arial') 
    fig = plt.figure(figsize=(1.4,0.9)) 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.plot(coords_M_raw[:,0], coords_M_raw[:,1],color="blue", marker='', linewidth=0.1)
    plt.scatter(coords_M_raw[:,0], coords_M_raw[:,1], s=0.15, c=blues, label='MAMMAL')
    x = np.ma.masked_where(coords_T_raw[:,0] == 0,coords_T_raw[:,0])
    y = np.ma.masked_where(coords_T_raw[:,1] == 0,coords_T_raw[:,1])

    plt.plot(x, y,color='red', marker='',linewidth=0.1)
    plt.scatter(x,y,s=0.15, c=reds,  label='Tri')
    plt.scatter(coords_G[:,0], coords_G[:,1], zorder=5, color='black', marker='^', s=1.2, label="Ground Truth")
    plt.xlim(-0.4,1.1)
    plt.ylim(-1.0,-0.1)
    plt.xticks([0, 1], ["0", "1"], fontsize=7)
    plt.yticks([-0.5,-1], ["-0.5", "-1"], fontsize=7)
    plt.xlabel("x (m)", fontsize=7) 
    plt.ylabel("y (m)", fontsize=7)
    ax = fig.get_axes()[0] 
    for line in ["bottom", "left", "right", "top"]: 
        ax.spines[line].set_linewidth(0.5)
    for line in ["top","right"]:
        ax.spines[line].set_visible(False)

    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    ax.plot(1.0, -1.0, ">k", markersize=0.8, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(-0.4, 1, "^k", markersize=0.8, transform=ax.get_xaxis_transform(), clip_on=False)

    plt.savefig("figs/Fig.3j.xy.png", dpi=1000, pad_inches=0.01, bbox_inches='tight')
    plt.savefig("figs/Fig.3j.xy.svg", dpi=1000, pad_inches=0.01, bbox_inches='tight')


def draw_z_shoulder():
    jointid = 6
    pid = 0

    R = np.arange(1100,1500)
    R_gt = np.arange(44,60)
    coords_M_raw = MAMMAL[R,pid,jointid]
    coords_T_raw = Tri[R,pid,jointid]
    coords_G = Gt[R_gt,pid,jointid]

    ### begin generating excel. 
    # import pandas as pd 
    # gt_x = []
    # gt_y = [] 
    # gt_z = [] 
    # for k in range(400): 
    #     if k % 25 == 0: 
    #         gt_x.append(coords_G[k//25, 0])
    #         gt_y.append(coords_G[k//25, 1])
    #         gt_z.append(coords_G[k//25, 2])
    #     else: 
    #         gt_x.append(None )
    #         gt_y.append(None )
    #         gt_z.append(None )
    
    # data_dict = { 
    #     "time (s)": (R / 25).tolist(),
    #     "GT (x)": gt_x,
    #     "MAMMAL (x)": coords_M_raw[:,0],
    #     "Tri (x)": coords_T_raw[:,0],
    #     "GT (y)": gt_y,
    #     "MAMMAL (y)": coords_M_raw[:,1],
    #     "Tri (y)": coords_T_raw[:,1],
    #     "GT (z)": gt_z,
    #     "MAMMAL (z)": coords_M_raw[:,2],
    #     "Tri (z)": coords_T_raw[:,2],
    # }
    # data = pd.DataFrame(data = data_dict) 
    # data.to_excel("excel/Fig3j.xlsx", sheet_name="data")
    # exit()
    ### end. 
    L = R.shape[0]
    blues = cm.get_cmap(plt.get_cmap('Blues'))(np.linspace(0.0,1.0,L))[:,:3]
    reds = cm.get_cmap(plt.get_cmap('Reds'))(np.linspace(0.0,1.0,L))[:,:3]

    mpl.rc('font', family='Arial') 
    fig = plt.figure(figsize=(1.4,0.9)) 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.plot(R/25, coords_M_raw[:,2], color='blue', linewidth=0.1, markersize=0)
    plt.scatter(R/25, coords_M_raw[:,2], color=blues, s=0.18)
    z = np.ma.masked_where(coords_T_raw[:,2] < 0.01, coords_T_raw[:,2])
    plt.plot(R/25, z, color='red', linewidth=0.1, markersize=0)
    plt.scatter(R/25, z, color=reds, s=0.18)
    plt.scatter(R_gt, coords_G[:,2], color='black', s=1.2, marker='^', zorder=5)
    plt.xlim(44,60)
    plt.ylim(0.1, 0.3)
    plt.xticks([44, 52, 60], fontsize=7)
    plt.yticks([0.1, 0.2, 0.3], fontsize=7)
    plt.xlabel("Time (s)", fontsize=7)
    plt.ylabel("z (m)", fontsize=7)
    ax = fig.get_axes()[0] 
    for line in ["bottom", "left", "right", "top"]: 
        ax.spines[line].set_linewidth(0.5)
    for line in ["top","right"]:
        ax.spines[line].set_visible(False)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    plt.savefig("figs/Fig.3j.z.png", dpi=1000, pad_inches=0.01, bbox_inches='tight')
    plt.savefig("figs/Fig.3j.z.svg", dpi=1000, pad_inches=0.01, bbox_inches='tight')


if __name__ == "__main__":
    if not os.path.exists("figs"):
        os.makedirs("figs") 

    draw_xy_shoulder() 
    draw_z_shoulder() 