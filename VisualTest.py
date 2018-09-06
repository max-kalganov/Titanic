from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from LogReg_class import LogisticRegressionModel


def create_dataSet():
    dataSet = np.random.random((15,2))

    kmeans = KMeans(n_clusters=2, max_iter = 10000,random_state=0).fit(dataSet)
    answerSet = kmeans.labels_

    print(dataSet)
    print(answerSet)

    dataSet_1 = dataSet[answerSet[:] == 1]
    dataSet_0 = dataSet[answerSet[:] == 0]

    plt.plot(dataSet_1, "xr")
    plt.plot(dataSet_0, "ob")
    plt.show()

dataSet = np.array([[1., 0.08518521,0.45985378],
                    [1., 0.13851827, 0.9419858 ],
                    [1., 0.30878526, 0.20883611],
                    [1., 0.0160247,  0.92335785],
                    [1., 0.74256369, 0.3699414 ],
                    [1., 0.54611513, 0.81464114],
                    [1., 0.8134179,  0.59915756],
                    [1., 0.76095869, 0.95217406],
                    [1., 0.05084209, 0.3055774 ],
                    [1., 0.02228888, 0.62681387],
                    [1., 0.51587439, 0.7808052 ],
                    [1., 0.72270179, 0.48777485],
                    [1., 0.81538251, 0.89336511],
                    [1., 0.21282806, 0.08611013],
                    [1., 0.22962954, 0.19158838]])

answerSet = np.array([[1], [1], [1], [1], [0], [0], [0], [0], [1], [1], [0], [0], [0], [1], [1]])


def train():
    obj = LogisticRegressionModel((dataSet.transpose(), answerSet), border=0.1, mode="d")
    obj.train()


train()


#TODO: Add plotting Cost func:
'''from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
a = np.random.random(7)
b = np.random.random(7)
c = np.random.random(7)
ax.scatter(a,b,c,"xr")
plt.show()'''