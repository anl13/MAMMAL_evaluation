import numpy as np 
import os 
from utils import * 
import pickle 
##
## Liang An 2022.07.20 
## To make the data more accessible, I pack raw joint data (without smoothing, used for paper evaluation figures)
## into pickle files. 
## 
def pack_to_pickle():
    result_folders = [
        ["H:/results/BamaPigEval3D_main/joints_23/", [0,2,3,1], "MAMMAL", 0],
        ["D:/results/paper_teaser/0704_eval2/joints_23/", [0,2,1,3], "MAMMAL_old", 750],
        ["E:/results/paper_teaser/0704_eval2-5views/joints_23/", [0,2,3,1], "MAMMAL(5)", 750],
        ["E:/results/paper_teaser/0704_eval2-(057)/joints_23/", [2,0,3,1], "MAMMAL(3)", 750],
        ["D:/results/paper_teaser/0704_eval_tri/skels/", [0,2,1,3], "Tri", 750],
        ["E:/results/paper_teaser/0704_eval2-5views/skels/", [0,2,3,1], "Tri(5)", 750],
        ["E:/results/paper_teaser/0704_eval2-(057)/skels/", [2,0,3,1], "Tri(3)", 750],
        # ["D:/results/paper_teaser/0704_eval_no_reassoc/joints_23/", [0,2,1,3]],
        # ["D:/results/paper_teaser/0704_eval_no_sil/joints_23/", [0,2,1,3]],
    ] 
    N = 70
    output_folder = "keypoints_for_eval" 
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder) 
    for result_config in result_folders: 
        est = load_joint23(result_config[0], start = result_config[3], step = 25, num=N, order = result_config[1]) 
        ## est is numpy asrray with shape [70, 4, 23, 3] in correct pig identity order
        outfile = os.path.join(output_folder, result_config[2] + ".pkl")
        with open(outfile, 'wb') as f: 
            pickle.dump(est, f)


if __name__ == "__main__": 
    pack_to_pickle() 