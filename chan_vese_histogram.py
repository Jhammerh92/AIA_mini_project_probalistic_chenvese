import numpy as np
import matplotlib.pyplot as plt
import AiA
import snake as snek
import cv2



# im = AiA.imread("probabilistic_data/overlap_test.png")
im = AiA.imread("probabilistic_data/plante_downsize.jpg", as_grey = True)
im = im*255
# AiA.imshow(im)

snake = snek.snake(100, im, tau=2.5, alpha=0.5,beta=0.02)

# snake.show() 
snake.init_EM_gaussians(peaks=3, std=35)
snake.plot_histograms(with_gaussians=True)
snake.EM_converge(iter=25)
snake.plot_histograms(with_gaussians=True)

plt.show()


# snake.converge_to_shape(ax=None, conv_lim_pix=0.01, show_normals=True)


# plt.show()