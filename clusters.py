#### Gustav Leth
#### Projekt Advanced Images Analysis
#### 21/04/2022

import numpy as np
import matplotlib.pyplot as plt
import snake as snek
import AiA
import cv2
from sklearn.cluster import KMeans
from PIL import Image


im= plt.imread("probabilistic_data/isbjorn_kpg.jpg") #RGB

snake = snek.snake(150, im,
                   tau=5,
                   var_tau=False,
                   alpha=0.1, 
                   beta=0.01, 
                   method="patch_prob", 
                   r=100,
                   weights=[1,1,1], # these are reweighted to sum to 1
                   patch_size=3,
                   n_dict=11,
                   n_clusters=7)
# print(snake.cluster_center_in)
# print(snake.cluster_center_out)

snake.show() 
snake.plot_histograms()

# snake.calc_clustering_histograms()
snake.plot_patch_dict()
snake.plot_patch_histograms()

snake.plot_cluster_dict()
snake.plot_cluster_histograms()

snake.plot_prob_maps()

# snake.plot_patch_dict()
snake.converge_to_shape(ax=None, show_normals=True, conv_lim_perc=0.001, max_iter= 120)


plt.show()

