import numpy as np
import pandas as pd
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
if __name__=='__main__':
    data, color = datasets.samples_generator.make_s_curve(1000, random_state=0)

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle('MDS and ISOMAP')

    mds2 = manifold.MDS(2, max_iter=100, n_init=1).fit_transform(data)
    ax = fig.add_subplot(222)
    plt.scatter(mds2[:,0],mds2[:,1],c=color)
    plt.title('MDS-2D')
    plt.axis('tight')

    lsomap2 = manifold.Isomap(10, 2).fit_transform(data)
    ax = fig.add_subplot(224)
    plt.scatter(lsomap2[:, 0], lsomap2[:, 1], c=color)
    plt.title('ISOMAP_2D')
    plt.axis('tight')

    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color)

    plt.title('original-data')
    plt.axis('tight')

    plt.show()
