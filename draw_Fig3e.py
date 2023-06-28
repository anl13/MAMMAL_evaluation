import numpy as np 
import pickle 
import json 
import os 
import cv2 

import seaborn as sns 
import pandas as pd 
from matplotlib.patches import Patch 
from matplotlib.lines import Line2D

import matplotlib as mpl 
import matplotlib.pyplot as plt 

def load_gt_small(): 
    # folder = "V:/rebuttal/pig_tail/sync_F2_batch1_test/labeled" 
    folder = "data/size/gt_small"
    out = [] 
    for index in range(12): 
        frameid = 12200 + index * 25
        filename = "joint3d_{:06d}.pkl".format(frameid) 
        with open(os.path.join(folder, filename), 'rb') as f: 
            data = pickle.load(f) 
            joints = data["joint3d"][0]
            out.append(joints)
    out = np.asarray(out) # 20, 2, 3, 3
    return out 

def load_gt_big(): 
    # folder = "V:/rebuttal/pig_tail/sync20210923_afternoon/labeled" 
    folder = "data/size/gt_big"
    out = [] 
    for index in range(12): 
        frameid = 24800 + index * 25
        filename = "joint3d_{:06d}.pkl".format(frameid) 
        with open(os.path.join(folder, filename), 'rb') as f: 
            data = pickle.load(f) 
            joints = data["joint3d"][0]
            out.append(joints)
    out = np.asarray(out) # 20, 2, 3, 3
    return out 

def load_gt_mid(): 
    # folder = "V:/rebuttal/pig_tail/sync20210923_afternoon/labeled"
    folder = "data/size/gt_mid"
    out = [] 
    for index in range(12): 
        frameid = 9500 + index * 25
        filename = "joint3d_{:06d}.pkl".format(frameid) 
        with open(os.path.join(folder, filename), 'rb') as f: 
            data = pickle.load(f) 
            joints = data["joint3d"][0]
            out.append(joints)
    out = np.asarray(out) # 20, 2, 3, 3
    return out 

def load_est_big(): 
    # folder = "D:/results_add/socialrank_0923a_notail/joints_23/"    
    folder = "data/size/est_big"
    est = np.zeros([12, 23, 3])
    for index in range(12): 
        frameid = 24800 + index * 25 
        joints = np.loadtxt(os.path.join(folder, "pig_{}_frame_{:06d}.txt".format(0, frameid))) 
        est[index] = joints 
    return est 

def load_est_small(): 
    # folder = "D:/results_add/F2_batch1_notail/joints_23/"    
    folder = "data/size/est_small"
    est = np.zeros([12, 23, 3])
    for index in range(12): 
        frameid = 12200 + index * 25
        joints = np.loadtxt(os.path.join(folder, "pig_{}_frame_{:06d}.txt".format(2, frameid))) 
        est[index] = joints 
    return est 

def load_est_mid(): 
    # folder = "D:/results_add/socialrank_0923a_notail2/joints_23/"
    folder = "data/size/est_mid"
    est = np.zeros([12, 23, 3])
    for index in range(12): 
        frameid = 9500 + index * 25 
        joints = np.loadtxt(os.path.join(folder, "pig_{}_frame_{:06d}.txt".format(1, frameid))) 
        est[index] = joints 
    return est 

joint_names = [
    "nose",
    "eye_left",
    "eye_right",
    "ear_root_left",
    "ear_root_right",
    "shoulder_left",
    "shoulder_right",
    "elbow_left",
    "elbow_right",
    "paw_left",
    "paw_right",
    "hip_left",
    "hip_right",
    "knee_left",
    "knee_right",
    "foot_left",
    "foot_right",
    "neck",
    "tail_root",
    "withers",
    "center",
    "tail_middle",
    "tail_end"
]


joint_names_body = [
    "Head",
    "Head",
    "Head",
    "Head",
    "Head",
    "FrontLeg",
    "FrontLeg",
    "FrontLeg",
    "FrontLeg",
    "FrontLeg",
    "FrontLeg",
    "HindLeg",
    "HindLeg",
    "HindLeg",
    "HindLeg",
    "HindLeg",
    "HindLeg",
    "Trunk",
    "Trunk",
    "Trunk",
    "Trunk",
    "Tail",
    "Tail"
]

def get_midsize_errors(): 
    with open("data/keypoints_for_eval/label_3d.pkl", 'rb') as f: 
        gts = pickle.load(f) 
        gt = gts[0:12,0,:,:]
    with open("data/keypoints_for_eval/MAMMAL.pkl", 'rb') as f: 
        ests = pickle.load(f) 
        est = ests[0:12,0,:,:]
    error = np.linalg.norm(gt - est, axis=-1) 
    return gt, est, error 
    
def build_data_frame(): 
    gt_good, est_good, error_good = get_midsize_errors() 

    gt_small = load_gt_small() 
    est_small = load_est_small() 
    
    gt_big = load_gt_big()
    est_big = load_est_big() 

    gt_mid = load_gt_mid() 
    est_mid = load_est_mid() 

    error_small = np.linalg.norm(gt_small - est_small, axis=-1) 
    error_big = np.linalg.norm(gt_big - est_big, axis=-1)
    error_mid = np.linalg.norm(gt_mid - est_mid, axis=-1) 

    error_list = [] 
    name_list  = [] 
    sizes = []

    error_good_list = [] 
    for k in range(error_good.shape[0]): 
        for j in range(23): 
            if np.linalg.norm(gt_good[k,j]) == 0: 
                continue 
            error_list.append(error_good[k, j] * 100) # cm 
            error_good_list.append(error_good[k,j] * 100)
            name_list.append(joint_names_body[j])
            sizes.append("good")
    print("good mean :", np.mean(error_good_list)) 
    print("good std  :", np.std(error_good_list) )


    error_big_list = [ ]
    for k in range(error_big.shape[0]): 
        for j in range(23): 
            if np.linalg.norm(gt_big[k,j]) == 0: 
                continue 
            error_list.append(error_big[k, j] * 100) # cm 
            error_big_list.append(error_big[k,j] * 100)
            name_list.append(joint_names_body[j])
            sizes.append("big")
    print("big  mean :", np.mean(error_big_list)) 
    print("big  std  :", np.std(error_big_list) )

    error_mid_list = []
    for k in range(error_mid.shape[0]): 
        for j in range(23): 
            if np.linalg.norm(gt_mid[k,j]) == 0: 
                continue 
            error_list.append(error_mid[k, j] * 100) # cm 
            error_mid_list.append(error_mid[k,j] * 100)
            name_list.append(joint_names_body[j])
            sizes.append("mid")
    print("mid  mean :", np.mean(error_mid_list)) 
    print("mid  std  :", np.std(error_mid_list) )

    error_small_list = [] 
    for k in range(error_small.shape[0]): 
        for j in range(23): 
            if np.linalg.norm(gt_small[k,j]) == 0: 
                continue 
            error_list.append(error_small[k, j] * 100) # cm 
            error_small_list.append(error_small[k,j] * 100)
            name_list.append(joint_names_body[j])
            sizes.append("small")
    print("small mean :", np.mean(error_small_list)) 
    print("small std  :", np.std(error_small_list) )

    d = { 
        "error": error_list, 
        "joints": name_list,
        "size": sizes
    }
    return pd.DataFrame(data = d)

def draw_error_figure(): 

    mpl.rc('font', family='Arial') 
    fig = plt.figure(figsize=(2.5,1.4)) 
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'in'

    color_maps = np.loadtxt("colormaps/anliang_paper.txt") / 255
    data_frame = build_data_frame() 

    ### generating excel 
    # data_frame = data_frame.replace("good", "Train Data")
    # data_frame = data_frame.replace("big", "Very Fat") 
    # data_frame = data_frame.replace("small", "Juvenile")
    # data_frame = data_frame.replace("mid", "Moderate")
    # data_frame.to_excel("excel/Fig3e.xlsx", sheet_name="data")
    # from IPython import embed; embed()
    # exit()
    ### end. 
    colors = {
        "good": np.asarray([1,1,1]),
        "big": np.asarray([0.7, 0.7, 0.7]),
        "mid": np.asarray([0.4, 0.4, 0.4]), 
        "small": np.asarray([0.1,0.1,0.1])
    }
    # order = ["TailRoot", "TailMid", "TailTip"] 

    ax = sns.boxplot(x="joints", y="error", hue = "size", data=data_frame, palette=colors,
        linewidth=0.5, sym="", showmeans=True,
        # order = order, 
        meanprops = {'marker':'s','markerfacecolor':'black','markeredgecolor':'black', 'linewidth':0, 'markersize':1}, 
        capprops={"linewidth":0.5, "color": 'k'},
        whiskerprops={"linewidth":0.5, "color": "k"},
        )
    # ax.grid(axis='y',color='gray', linestyle='--', linewidth=0.2, zorder=0) 
    
    # plt.xticks([0,1,2], ["Tail\nRoot", "Tail\nMid", "Tail\nTip"], rotation=0, ha='center', fontsize=7)
    legend_elements = [
        Patch(facecolor=colors["good"], edgecolor='black', label="Train Data", linewidth=0.5), 
        Patch(facecolor=colors["big"], edgecolor='black', label="Very Fat", linewidth=0.5), 
        Patch(facecolor=colors["mid"], edgecolor='black',label="Moderate", linewidth=0.5),
        Patch(facecolor=colors["small"], edgecolor='black',label="Juvenile", linewidth=0.5),
    ]
    plt.legend(handles=legend_elements, fontsize=7, ncol=2, frameon=False)
    plt.xticks(rotation=0, ha='center', fontsize=7)
    ax = fig.get_axes()[0]
    for line in ["bottom", "left", "right", "top"]: 
        ax.spines[line].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    
    plt.xlabel("", fontsize=7)
    plt.ylabel("Error (cm)", fontsize=7)
    plt.ylim(0,20)
    plt.yticks([0,5,10,15], labels=[0, 5, 10, 15], fontsize=7)
    plt.savefig("figs/Fig.3e.png", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    plt.savefig("figs/Fig.3e.svg", dpi=1000, bbox_inches='tight', pad_inches=0.01) # uncomment this to write vector image.


if __name__ == "__main__": 
    draw_error_figure()

 