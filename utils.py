import numpy as np 
import os 

g_jointnames = [
    "nose", 
    "l_eye", 
    "r_eye", 
    "l_ear", 
    "r_ear", 
    "l_shoulder", 
    "r_shoulder", 
    "l_elbow", 
    "r_elbow", 
    "l_paw", 
    "r_paw", 
    "l_hip", 
    "r_hip", 
    "l_knee", 
    "r_knee", 
    "l_foot", 
    "r_foot", 
    "none", 
    "tail", 
    "none", 
    "center", 
    "none", 
    "none"
]

g_groupnames = [
    "Head", 
    "Head", 
    "Head", 
    "Head", 
    "Head", 
    "L_arm", 
    "R_arm", 
    "L_arm", 
    "R_arm", 
    "L_arm", 
    "R_arm", 
    "L_leg", 
    "R_leg", 
    "L_leg", 
    "R_leg", 
    "L_leg", 
    "R_leg", 
    "none", 
    "Tail", 
    "none", 
    "Center", 
    "none", 
    "none"
]

g_all_parts = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20]
g_head = [0,1,2,3,4]
g_left_front_leg = [5,7,9]
g_right_front_leg = [6,8,10]
g_left_hind_leg = [11,13,15]
g_right_hind_leg = [12,14,16]
g_legs = g_left_front_leg + g_left_hind_leg + g_right_front_leg + g_right_hind_leg
g_leg_level1 = [5,6,11,12]
g_leg_level2 = [7,8,13,14]
g_leg_level3 = [9,10,15,16]
g_trunk = [20,18]
g_pig_ids_for_eval = [0,1,2,3]


def load_joint23(folder, start, step = 25, num = 70, order=[0,1,2,3]):
    all_data = [] 
    for i in range(num):
        frameid = start + step * i 
        single_frame = [0,1,2,3] 
        for pid in range(4):
            filename = folder + "/pig_{}_frame_{:06d}.txt".format(pid, frameid)
            if not os.path.exists(filename): 
                filename = folder + "/pig_{}_{:06d}.txt".format(pid, frameid)
            data = np.loadtxt(filename) 
            index = order[pid]
            single_frame[index] = data 
        all_data.append(single_frame) 
    all_data = np.asarray(all_data) 

    return all_data 
