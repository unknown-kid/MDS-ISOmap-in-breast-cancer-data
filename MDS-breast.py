import numpy as np
import pandas as pd
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
#多维空间计算欧式距离的函数
def distance(a,b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i]-b[i])**2
    return math.sqrt(sum)


def classify(data):
    optimum = []
    malignant = []
    for i in data:
        if i[9] == 2:
            del(i[9])
            optimum.append(i)
        else:
            del(i[9])
            malignant.append(i)
    return optimum,malignant
#主函数
if __name__=='__main__':
    #获取bsd.txt中的乳腺癌数据
    data = []
    with open('bsd.txt','r') as f:
        for line in f:
            a = line[0:-1].split(',')
            del(a[0])
            b = []
            for i in a:
                if i =='?':
                    break
                else:
                    b.append(int(i))
            if len(b)==10:
                data.append(b)
    '''
    #把数据转换成浮点型的numpy数组，方便后面的计算
    data1 = np.array(data, dtype=np.float32)
    #调用ISOmap函数进行维数约简为8维，然后把data数据喂进去
    lsomap9 = manifold.Isomap(500, 8).fit_transform(data1)
    #调用MDS函数进行维数约简为8维，然后把data数据喂进去
    mds9 = manifold.MDS(8, max_iter=100, n_init=1).fit_transform(data1)
    #通过函数进行计算各自的欧式距离
    d10 = [distance(data1[i],data1[i+1]) for i in range(len(data1)-1)]
    di9 = [distance(lsomap9[i], lsomap9[i + 1]) for i in range(len(lsomap9) - 1)]
    dm9 = [distance(mds9[i], mds9[i + 1]) for i in range(len(mds9) - 1)]
    #打印数据
    for i in range(10):
        print('%.5f\t%.5f\t%.5f\n'%(d10[i],dm9[i],di9[i]))
    '''


    # 把数据分类成良性和恶性两种
    optimum,malignant = classify(data)
    optimum = np.array(optimum, dtype=np.float32)
    malignant = np.array(malignant, dtype=np.float32)
    #打开画布，大小设置为15,8
    fig = plt.figure(figsize=(15, 8))
    #设置大标题
    fig.suptitle('breast cancer data in MDS and ISOMAP')
    mds2o = manifold.MDS(2, max_iter=100, n_init=1).fit_transform(optimum)
    mds2m = manifold.MDS(2, max_iter=100, n_init=1).fit_transform(malignant)
    #设置画布分成2*2的块，ax分配第一块
    ax = fig.add_subplot(221)
    #把MDS约简后的数据散点显示在画布的第一块
    ax.scatter(mds2o[:,0],mds2o[:,1],c='red',marker='+')
    ax.scatter(mds2m[:, 0], mds2m[:, 1], c='blue',marker='>')
    #打印第一块的标题
    plt.title('MDS-2D')
    #使得图形变换大小，把所有数据都能够显示出来
    plt.axis('tight')

    lsomap2o = manifold.Isomap(10, 2).fit_transform(optimum)
    lsomap2m = manifold.Isomap(10, 2).fit_transform(malignant)
    ax = fig.add_subplot(222)
    ax.scatter(lsomap2o[:, 0], lsomap2o[:, 1], c='red',marker='+')
    ax.scatter(lsomap2m[:, 0], lsomap2m[:, 1], c='blue',marker='>')
    plt.title('ISOMAP_2D')
    plt.axis('tight')

    mds3o = manifold.MDS(3, max_iter=100, n_init=1).fit_transform(optimum)
    mds3m = manifold.MDS(3, max_iter=100, n_init=1).fit_transform(malignant)
    #后面的projection是表示画布用三维的表示
    ax = fig.add_subplot(223, projection='3d')
    ax.scatter(mds3o[:, 0], mds3o[:, 1], mds3o[:, 2], c='red',marker='+')
    ax.scatter(mds3m[:, 0], mds3m[:, 1], mds3m[:, 2], c='blue',marker='>')
    plt.title('MDS_3D')
    plt.axis('tight')

    lsomap3o = manifold.Isomap(10, 3).fit_transform(optimum)
    lsomap3m = manifold.Isomap(10, 3).fit_transform(malignant)
    ax = fig.add_subplot(224, projection='3d')
    ax.scatter(lsomap3o[:, 0], lsomap3o[:, 1], lsomap3o[:, 2], c='red',marker='+')
    ax.scatter(lsomap3m[:, 0], lsomap3m[:, 1], lsomap3m[:, 2], c='blue', marker='>')
    plt.title('lsomap_3D')
    plt.axis('tight')
    #最后把画布给展示出来
    plt.show()