from re import A
import matplotlib.pyplot as plt 
import numpy as np 
import pickle 
import json 
import os
from mpl_toolkits import mplot3d
from scipy.stats import sem 
from matplotlib import cm 
import matplotlib as mpl 
import cv2 

from OpenGL.GL import *
import glfw
from glfw.GLFW import *
from numpy.lib.twodim_base import fliplr
from pig_render.common import *
from pig_render.Render import *
from pig_render import MainRender
from pig_render.Render import OBJ
import glm 
from utils import g_all_parts, g_jointnames

'''
ATTENTION! 
This file render 3D skeleton of both MAMMAL result, Tri result and Ground Truth labels. 
The ground truth label is the mixed label as explained in Supplementary Fig. 8. 

Black cube joint: GT 
Blue  ball joint: MAMMAL 
Red   ball joint: Tri 

While rendering, press ECS to exit, press 's' to save render result. 
'''
g_bones_19 = [ # bone structure for the final 19 valid keypoints  
    [0,1],
    [0,2],
    [1,2],
    [1,3], 
    [2,4],
    [0,18], 
    [18,17],
    [18,5],
    [5,7],
    [7,9],
    [18,6],
    [6,8],
    [8,10],
    [17,11],
    [11,13], 
    [13,15],
    [17,12],
    [12,14],
    [14,16]
]


def draw_3D_demo():
    with open("data/traj/MAMMAL.pkl", 'rb') as f: 
        MAMMAL = pickle.load(f)  # 1750, 4, 23, 3
    with open("data/traj/Tri.pkl", 'rb') as f: 
        Tri = pickle.load(f)  # 1750, 4, 23, 3
    with open("data/keypoints_for_eval/label_mix.pkl", 'rb') as f: ### YOU can change label_mix to label_mesh or label_3d as you like. 
        Gt = pickle.load(f)  # 70, 4, 23,3
    MAMMAL = MAMMAL[:,:,g_all_parts, :]
    Tri = Tri[:,:,g_all_parts, :]
    Gt = Gt[:,:,g_all_parts, :]

    sample_ids = np.arange(0, 1750, 25)
    MAMMAL_sample = MAMMAL[sample_ids]
    Tri_sample = Tri[sample_ids]
    
    '''
    You can choose which frame to show. 
    0~69 are valid. (only 70 frames are labeled with 3d gt)
    69 is used for paper Fig. 3e 
    '''
    frameid_gt = 69

    renderer = MainRender(1920,1080)
    window = renderer.window
    mesh3 = MeshRender()  
    floormodel = OBJ("pig_render/data/obj_model/floor_z+_gray.obj")
    mesh3.load_data_basic(vertex = floormodel.vertices, face = floormodel.faces_vert)
    mesh3.load_data_colors(colors=floormodel.colors)
    mesh3.bind_arrays()
    renderer.mesh_object_list.append(mesh3) 

    colors = np.asarray([ 
        [8, 48, 107],
        [103,0,13],
        [1,1,1]
    ]) / 110 
    ## render MAMMAL  
    for pid in range(4):
        joints = MAMMAL_sample[frameid_gt,pid]
        skel = SkelRender(jointnum=joints.shape[0], topo = g_bones_19) 
        skel.set_joint_position(joints) 
        skel.set_joint_bone_color(colors[0], np.asarray(colors[0]))
        skel.set_ball_size(0.008)
        skel.set_stick_size(0.003)
        skel.bind_arrays() 
        skel.isFill = True
        renderer.skel_object_list.append(skel) 

    ## render triangulation  
    for pid in range(4):
        joints = Tri_sample[frameid_gt,pid]
        skel = SkelRender(jointnum=joints.shape[0], topo = g_bones_19) 
        skel.set_joint_position(joints) 
        skel.set_joint_bone_color(colors[1], np.asarray(colors[1]))
        skel.set_ball_size(0.008)
        skel.set_stick_size(0.003)
        skel.bind_arrays() 
        skel.isShowZeroPoint = False
        skel.isUseCube = False 
        skel.isFill = True
        renderer.skel_object_list.append(skel) 

    ## render gt 
    for pid in range(4):
        joints = Gt[frameid_gt,pid]
        skel = SkelRender(jointnum=joints.shape[0], topo = g_bones_19) 
        skel.isUseCube = True 
        skel.set_joint_position(joints) 
        skel.set_joint_bone_color(colors[2], colors[2])
        skel.set_ball_size(0.008)
        skel.set_stick_size(0.003)
        skel.bind_arrays() 
        skel.isShowZeroPoint = False
        renderer.skel_object_list.append(skel) 

    pos = [ -1.5233031613510213 , 1.7125846899443342 , 2.767258308059608 ]
    up = [ 0.4598253773469926 , -0.5393001996888378 , 0.7054898418569014 ]
    cen = [ 0.32879436 , -0.1226489 , 0.17539386 ]
    pos = np.asarray(pos) 
    up = np.asarray(up) 
    cen = np.asarray(cen)
    g_renderCamera.computeExtrinsic(pos,up,cen)
    g_mouseState["print_cam_pos"] = False 
    renderer.is_use_background_img = False 
    while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS and glfwWindowShouldClose(window) == 0):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        renderer.set_bg_color((1,1,1))
        renderer.draw() 

        if g_mouseState["saveimg"]: 
            img = renderer.readImage() 
            cv2.imwrite("saveimg{}.jpg".format(g_mouseState["saveimg_index"]), img) 
            g_mouseState["saveimg"] = False 
            g_mouseState["saveimg_index"] += 1
  
        glfwSwapBuffers(window)
        glfwPollEvents()

    glfwTerminate()

if __name__ == "__main__":
    draw_3D_demo() 