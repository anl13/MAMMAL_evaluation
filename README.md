# MAMMAL Evaluation 
This repository presents how to produce evaluation figures (Fig. 2b-f) proposed in paper: 

An, L., Ren, J., Yu, T., Hai, T., Jia, Y., &Liu, Y. Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL. *biorxiv* (2022).

[ [project]() ] [ [paper]() ]

Other related repositories: 
* [MAMMAL_core]() 
* [MAMMAL_datasets](https://github.com/anl13/MAMMAL_datasets) 
* [MAMMAL_behavior](https://github.com/anl13/MAMMAL_behavior) 
* [pig_silhouette_det](https://github.com/anl13/pig_silhouette_det)
* [pig_pose_det](https://github.com/anl13/pig_pose_det)
* [PIG_model](https://github.com/anl13/PIG_model) 

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
`-- draw_Fig2b.py
`-- ...
```
4. Run drawing codes by 
```
python draw_Fig2b.py
python draw_Fig2c.py
python draw_Fig2d.py
python draw_Fig2e.py
python draw_Fig2f.py
```
and you will get two output folders: (1) `data_for_fig/` saves the middle data; (2) `figs/` saves the output figures same to our paper. 

## Citation
If you use these datasets in your research, please cite the paper

```BibTex
@article{MAMMAL, 
    author = {An, Liang and Ren, Jilong and Yu, Tao and Jia, Yichang and Liu, Yebin},
    title = {Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL},
    booktitle = {biorxiv},
    month = {July},
    year = {2022}
}
```

## Contact
* Liang An ([anl13@mail.tsinghua.org.cn](anl13@mail.tsinghua.org.cn))
* Yebin Liu ([liuyebin@mail.tsinghua.edu.cn](liuyebin@mail.tsinghua.edu.cn))