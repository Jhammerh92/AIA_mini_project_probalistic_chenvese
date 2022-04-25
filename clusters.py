#### Gustav Leth
#### Projekt Advanced Images Analysis
#### 21/04/2022

import numpy as np
import matplotlib.pyplot as plt
import os
import AiA
import cv2
from sklearn.cluster import KMeans
from PIL import Image


fisk = plt.imread("probabilistic_data/simple_test.png") #RGB


print("hej")

def clustering(image_path, num_clusters):
    image = plt.imread(image_path)
    # red  = image[:,:,0]
    # green = image[:,:,1]
    # blue  = image[:,:,2]
    Z = np.reshape(image, (-1, 3))
    
    name = "K-means"
    # define criteria and apply kmeans()
    model = KMeans(n_clusters=num_clusters).fit(Z)
    label = model.labels_
    # center = kmeans.cluster_centers_
       
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # ret,label,center=cv.kmeans(Z,num_clusters,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    return model, label, Z


def plot_clustering_2D(image_path, num_clusters):

    model, label, Z = clustering(image_path, num_clusters)
    center = model.cluster_centers_

    Cluster_A = Z[label.ravel()==0]
    Cluster_B = Z[label.ravel()==1]
    # Cluster_C = Z[label.ravel()==2]

    # Plot the data
    fig, axes = plt.subplots(1,3)

    # plt.figure("Red / Green")
    axes[0].scatter(Cluster_A[:,0],Cluster_A[:,1], c = 'b')
    axes[0].scatter(Cluster_B[:,0],Cluster_B[:,1], c = 'r')
    axes[0].scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    axes[0].set_xlabel("red"), axes[0].set_ylabel('Green')

    # plt.figure("Red / Blue")
    axes[1].scatter(Cluster_A[:,0],Cluster_A[:,2], c = 'b')
    axes[1].scatter(Cluster_B[:,0],Cluster_B[:,2], c = 'r')
    axes[1].scatter(center[:,0],center[:,2],s = 80,c = 'y', marker = 's')
    axes[1].set_xlabel("red"), axes[1].set_ylabel("blue")

    # plt.figure("Green / Blue")
    axes[2].scatter(Cluster_A[:,1],Cluster_A[:,2], c = 'b')
    axes[2].scatter(Cluster_B[:,1],Cluster_B[:,2], c = 'r')
    axes[2].scatter(center[:,1],center[:,2],s = 80,c = 'y', marker = 's')
    axes[2].set_xlabel("green"), axes[2].set_ylabel("blue")

    plt.show()

    return


def plot_clustering_3D(image_path, num_clusters):

    model, label, Z = clustering(image_path, num_clusters)
    center = model.cluster_centers_

    Cluster_A = Z[label.ravel()==0]
    Cluster_B = Z[label.ravel()==1]
    # Cluster_C = Z[label.ravel()==2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
        xs = Cluster_A[:,0]
        ys = Cluster_A[:,1]
        zs = Cluster_A[:,2]
        ax.scatter(xs, ys, zs, marker='.', color = 'b')
        xs = Cluster_B[:,0]
        ys = Cluster_B[:,1]
        zs = Cluster_B[:,2]
        ax.scatter(xs, ys, zs, marker='.', color = 'r')
        # xs = Cluster_C[:,0]
        # ys = Cluster_C[:,1]
        # zs = Cluster_C[:,2]
        # ax.scatter(xs, ys, zs, marker='.', color = 'g')

        cenX = center[:,0]
        cenY = center[:,1]
        cenZ = center[:,2]
        ax.scatter(cenX, cenY, cenZ, marker='o', s = 100, color = 'y')
        
        
    ax.set_xlabel('RED')
    ax.set_ylabel('GREEN')
    ax.set_zlabel('BLUE')
    
    plt.show()
    
    return

# plot_clustering_3D("probabilistic_data/plante_downsize.jpg", 2)
# plot_clustering_2D("probabilistic_data/plante_downsize.jpg", 2)
