import numpy as np
import matplotlib.pyplot as plt
import AiA
import snake as snek
import cv2



im = AiA.imread("probabilistic_data/simple_test.png", as_type=True,load_type=np.uint8)
# im = AiA.imread("probabilistic_data/plante_downsize.jpg", as_grey = True)
im = im
# AiA.imshow(im)

snake = snek.snake(100, im, tau=2.5, alpha=0.5,beta=0.02)

snake.show() 
snake.init_EM_gaussians(peaks=3, std=35)
snake.plot_histograms(with_gaussians=False)
# snake.EM_converge(iter=100)
# snake.plot_histograms(with_gaussians=False)
plt.show()


# snake.converge_to_shape(ax=None, conv_lim_pix=0.01, show_normals=True)


# plt.show()