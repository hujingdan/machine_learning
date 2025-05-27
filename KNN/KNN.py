from sklearn import neighbors,datasets
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import numpy as np

#建立KNN模型，使用前两个特征
##调用鸢尾花数据集赋值iris
iris = datasets.load_iris()
##抽取数据集中前两个特征
irisData = iris.data[:, :2]
##再取出鸢尾花类别标签
irisTarget = iris.target
##建立KNN模型
knn = neighbors.KNeighborsClassifier(n_neighbors=5) #K=5
##训练模型
knn.fit(irisData, irisTarget)

#绘制plot
##设置颜色
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
##设置网格
x_min, x_max = irisData[:, 0].min() , irisData[:, 0].max() 
y_min, y_max = irisData[:, 1].min() , irisData[:, 1].max() 
##设置网格点
X,Y = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

#预测
##预测网格点上的值
Z = knn.predict(np.c_[X.ravel(), Y.ravel()])
##将预测值转换为网格
Z = Z.reshape(X.shape)

#绘图并显示
plt.figure()
plt.pcolormesh(X, Y, Z, cmap=cmap_light)
plt.show()


