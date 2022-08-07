import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import pandas as pd 
from matplotlib.patches import Patch 
from matplotlib.patches import ConnectionPatch
import os 
from draw_Fig3g import compute_loss 
import pickle 
from utils import g_all_parts 

def load_and_eval(configname, N):
    with open("data/keypoints_for_eval/label_mix.pkl", 'rb') as f: 
        gt = pickle.load(f) 
    with open("data/keypoints_for_eval/" + configname + ".pkl", 'rb') as f: 
        est = pickle.load(f) 
    compute_loss(est, gt, N=N, name=configname, part=g_all_parts, part_names="all_parts_mix") 

# last modified: 
# 2021.10.21 by An Liang
# draw main error figure 
def main_error(): 
    mpl.rc('font', family='Arial') 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['figure.autolayout'] = True
    fig, (ax, ax1) = plt.subplots(1,2,figsize=(3.4,1.7)) 

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

    '''
    ATTENTION: 
    Here relies on the data from draw_Fig2f.py
    '''
    folder = "data_for_fig/"
    data_MAMMAL_10 = np.loadtxt(folder + "MAMMAL_all_parts_mix.txt")
    data_MAMMAL_5 = np.loadtxt(folder +"MAMMAL(5)_all_parts_mix.txt")
    data_MAMMAL_3 = np.loadtxt(folder +"MAMMAL(3)_all_parts_mix.txt")
    data_Tri_10 = np.loadtxt(folder +"Tri_all_parts_mix.txt")
    data_Tri_5 = np.loadtxt(folder +"Tri(5)_all_parts_mix.txt")
    data_Tri_3 = np.loadtxt(folder +"Tri(3)_all_parts_mix.txt")

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

    avg = ["3.44","4.08","5.19","14.17", "24.19", "41.81"]
    
    bplot = ax.boxplot(x = all_data, # 指定绘图数据
                sym = "", 
                patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
                showmeans=True, # 以点的形式显示均值
                boxprops = {'color':'black','facecolor':'#9999ff','linewidth':0.5}, # 设置箱体属性，填充色和边框色
                flierprops = {'marker':'o','markerfacecolor':'red','color':'black','linewidth':0.5}, # 设置异常值属性，点的形状、填充色和边框色
                meanprops = {'marker':'s','markerfacecolor':'black','markeredgecolor':'black', 'linewidth':0, 'markersize':1.5}, # 设置均值点的属性，点的形状、填充色
                medianprops = {'linestyle':'--','color':'black','linewidth':0.5},
                capprops={"linewidth":0.5},
                whiskerprops={"linewidth":0.5},
                labels = labels, 
                ) # 设置中位数线的属性，线的类型和颜色
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color) 
        patch.set_linewidth(0.5)
    ax.plot([0,7],[0.2,0.2],linewidth=0.5,linestyle='--',color='k')
    ax.plot([7,7.5],[0.2,1.8],linewidth=0.5, linestyle='--',color='k')
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='black',label=labels[0], linewidth=0.5), 
        Patch(facecolor=colors[1], edgecolor='black',label=labels[1], linewidth=0.5), 
        Patch(facecolor=colors[2], edgecolor='black',label=labels[2], linewidth=0.5), 
        Patch(facecolor=colors[3], edgecolor='black',label=labels[3], linewidth=0.5), 
        Patch(facecolor=colors[4], edgecolor='black',label=labels[4], linewidth=0.5), 
        Patch(facecolor=colors[5], edgecolor='black',label=labels[5], linewidth=0.5), 
    ]
    ax.legend(handles=legend_elements, fontsize=6, frameon=False)
    ax = fig.get_axes()[0]
    for line in ["bottom", "left", "right", "top"]: 
        ax.spines[line].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)


    # 设置y轴的范围
    ax.set_ylim(0, 1.8)
    ax.set_xlim(0.5,6.5)
    ax.set_xticks([])
    ax.set_yticks([0,0.4,0.8,1.2,1.6])
    ax.set_yticklabels(labels=[0,40,80,120,160], fontsize=7, family='Arial')
    ax.text(1-0.25,0.26,avg[0],fontsize=4)
    ax.text(2-0.25,0.26,avg[1],fontsize=4)
    ax.text(3-0.25,0.26,avg[2],fontsize=4)
    ax.text(4-0.35,0.26,avg[3],fontsize=4)
    ax.text(5-0.35,0.44,avg[4],fontsize=4)
    ax.text(6-0.35,1.62,avg[5],fontsize=4)

    ax.set_ylabel("Error (cm)", fontsize=7, family='Arial')


    bplot = ax1.boxplot(x = all_data, # 指定绘图数据
                sym = "", 
                patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
                showmeans=True, # 以点的形式显示均值
                boxprops = {'color':'black','facecolor':'#9999ff','linewidth':0.5}, # 设置箱体属性，填充色和边框色
                flierprops = {'marker':'o','markerfacecolor':'red','color':'black','linewidth':0.5}, # 设置异常值属性，点的形状、填充色和边框色
                meanprops = {'marker':'s','markerfacecolor':'black','markeredgecolor':'black', 'linewidth':0, 'markersize':1.5}, # 设置均值点的属性，点的形状、填充色
                medianprops = {'linestyle':'--','color':'black','linewidth':0.5},
                capprops={"linewidth":0.5},
                whiskerprops={"linewidth":0.5},
                labels = labels, 
                ) # 设置中位数线的属性，线的类型和颜色
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color) 
        patch.set_linewidth(0.5)
    # plt.grid(linestyle="--", alpha=0.3)

    # ax.spines['top'].set_visible(False)
    # ax.spines["right"].set_visible(False)
    for line in ["bottom", "left", "right", "top"]: 
        ax1.spines[line].set_linewidth(0.5)
    ax1.xaxis.set_tick_params(width=0.5)
    ax1.yaxis.set_tick_params(width=0.5)
    ax1.yaxis.tick_right()
    # 设置y轴的范围
    ax1.set_ylim(0, 0.2)
    ax1.set_xlim(0.5,6.5)
    # plt.xticks(fontsize=20, family='Arial')
    ax1.set_xticks([])
    ax1.set_yticks([0,0.1,0.2])
    ax1.set_yticklabels(labels=[0,10,20],fontsize=7)

    con = ConnectionPatch([6.5,0.2], [0.5,0.2], coordsA=ax.transData, coordsB=ax1.transData,
            linewidth=0.5,linestyle='--')
    con1 = ConnectionPatch([6.5,0], [0.5,0], coordsA=ax.transData, coordsB=ax1.transData,
            linewidth=0.5,linestyle='--')
    fig.add_artist(con)
    fig.add_artist(con1)

    plt.savefig("figs/Fig.3d.png", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    # plt.savefig("figs/Fig.3d.svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)

if __name__ == "__main__":
    if not os.path.exists("figs"):
        os.makedirs("figs")
    main_error()
