import numpy as np
import matplotlib.pyplot as plt
import AiA
import snake as snek
import cv2



# im = AiA.imread("probabilistic_data/color_test_01.png", as_type=True,load_type=np.uint8)
# im = AiA.imread("probabilistic_data/textured_test.png", as_type=True,load_type=np.uint8)
im = AiA.imread("probabilistic_data/134052.jpg", as_type=True,load_type=np.uint8)
# im = AiA.imread("probabilistic_data/isbjorn.jpg", as_type=True,load_type=np.uint8)
# im = AiA.imread("probabilistic_data/plante_downsize.jpg")
# im = im*255
# AiA.imshow(im)

snake = snek.snake(150, im,
                   tau=10,
                   var_tau=False,
                   alpha=0.01, 
                   beta=0.01, 
                   method="patch_prob", 
                   r=150,
                   weights=[0.0,0.65,0.35], 
                   patch_size=10,
                   n_dict=20)

snake.show() 
# snake.init_EM_gaussians(peaks=3, std=35)
snake.plot_histograms(with_gaussians=False)
# snake.init_patch_dict(patch_size=11)

# snake.calc_patch_knn()

snake.plot_patch_dict()
snake.plot_patch_histograms()
snake.plot_prob_maps()



# snake.plot_histograms(with_gaussians=False)
# plt.show()

snake.converge_to_shape(ax=None, show_normals=True, min_avg=40, min_iter=30, conv_lim_perc=1e-5)
# snake.plot_patch_dict()
snake.plot_patch_histograms()
snake.plot_prob_maps()
snake.show() 

plt.show()

