# MAMMAL Evaluation 
This repository presents how to produce evaluation figures (Fig. 3) proposed in [MAMMAL](https://github.com/anl13/MAMMAL_core).

## How to run the codes  
These codes are written in Python and work under both windows 10 and Ubuntu (>=16.04). 
1. Install conda. (follow official anaconda tutorial)
2. Create conda virtual environment and install required packages.  
```shell 
conda create -n MAMMAL_eval python=3.7
conda activate MAMMAL_eval
pip install -r requirements.txt
```
3. Unzip `data.zip` and the file structure should looks like 
```
MAMMAL_evaluation/
`-- data
    |-- iou_for_eval/
    |-- keypoints_for_eval/
    |-- traj/
`-- draw_Fig2b.py
`-- ...
```
If `data/` folder already exists, then you can simply pass this step. 
4. Run drawing codes by 
```
python draw_Fig3b.py
python draw_Fig3c.py
python draw_Fig3e.py
python draw_Fig3f.py
python draw_Fig3g.py
python draw_Fig3h.py
python draw_Fig3i.py
python draw_Fig3j.py
```
and you will get two output folders: (1) `data_for_fig/` saves the middle data; (2) `figs/` saves the output figures same to our paper. 

If you want to run `draw_Fig3i.py`, you need to prepare `pig_render` like 
```
MAMMAL_evaluation/
`-- data
    |-- iou_for_eval/
    |-- keypoints_for_eval/
    |-- traj/
`-- pig_render/
`-- draw_Fig2b.py
`-- ...
```
`pig_render` can be downloaded from https://github.com/anl13/pig_renderer.
## Citation
If you use these datasets in your research, please cite the paper

```BibTex
@article{MAMMAL, 
    author = {An, Liang and Ren, Jilong and Yu, Tao and Hai, Tang and Jia, Yichang and Liu, Yebin},
    title = {Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL},
    booktitle = {},
    month = {July},
    year = {2022}
}
```