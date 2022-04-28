import numpy as np
import matplotlib.pyplot as plt
import AiA
import snake as snek
import cv2



# im = AiA.imread("probabilistic_data/color_test_01.png", as_type=True,load_type=np.uint8)
im = AiA.imread("probabilistic_data/134052.jpg", as_type=True,load_type=np.uint8)
# im = AiA.imread("probabilistic_data/plante_downsize.jpg", as_grey = True)
# im = im*255
# AiA.imshow(im)

snake = snek.snake(150, im,
                   tau=5,
                   alpha=0.01, 
                   beta=0.1, 
                   method="patch_prob", 
                   r=125, 
                   patch_size=5)

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

# snake.converge_to_shape(ax=None, conv_lim_pix=0.01, show_normals=True)
# snake.plot_patch_dict()
snake.plot_patch_histograms()

plt.show()

