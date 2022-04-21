import numpy as np
import matplotlib.pyplot as plt
import AiA
import snake as snek
import cv2



im = AiA.imread("probabilistic_data/overlap_test.png")

AiA.imshow(im)

snake = snek.snake(100, im, tau=2.5, alpha=0.5,beta=0.02)

snake.converge_to_shape(ax=None, conv_lim_pix=0.01, show_normals=True)


# plt.show()