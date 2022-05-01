import numpy as np
import matplotlib.pyplot as plt
import AiA
import snake as snek
import cv2



# im = plt.imread("probabilistic_data/william.jpg")
# im = plt.imread("probabilistic_data/color_test_01.png")
# im = AiA.imread("probabilistic_data/simple_test.png", as_type=True,load_type=np.uint8)
# im = AiA.imread("probabilistic_data/randen15.png", as_type=True,load_type=np.uint8)
# im = AiA.imread("probabilistic_data/test_C.png", as_type=True,load_type=np.uint8)
# im = AiA.imread("probabilistic_data/12003.jpg")
im = AiA.imread("probabilistic_data/134052.jpg")
# im = AiA.imread("probabilistic_data/105053.jpg")
# im = plt.imread("probabilistic_data/124084.jpg")
# im = plt.imread("probabilistic_data/108073.jpg")
# im = plt.imread("probabilistic_data/164074.jpg")
# im = AiA.imread("probabilistic_data/Mindre_isbjorn.jpg", as_type=True,load_type=np.uint8)
# im = AiA.imread("probabilistic_data/plante.jpg")
# im = AiA.imread("probabilistic_data/flower_petal_01.jpg")
# print(im)

# im = im*255
# AiA.imshow(im)

snake = snek.snake(150, im,
                   tau=3,
                   var_tau=False,
                   alpha=0.01, 
                   beta=0.01, 
                   method="unify", 
                   r=None,
                   weights=[1,10,10], # these are reweighted to sum to 1
                   patch_size=7,
                   n_dict=20,
                   n_clusters=5,
                   shift=[0,0])


# snake.init_snake_to_image(r=None,shift = shift) # shift from center

snake.show() 
snake.plot_histograms()

# snake.calc_clustering_histograms()
snake.plot_patch_dict()
snake.plot_patch_histograms()



snake.plot_cluster_dict()
snake.plot_cluster_histograms()


snake.plot_prob_maps()


# # plt.show()


snake.converge_to_shape(ax=None, show_normals=True, min_avg=40, min_iter=30, conv_lim_perc=1e-4)

# # snake.plot_patch_dict()
# snake.plot_patch_histograms()
# snake.plot_cluster_histograms()

# snake.plot_prob_maps()
# snake.show() 

plt.show()

