# import sys
# print('Python: {}'.format(sys.version))
# # scipy
# import scipy
# print('scipy: {}'.format(scipy.__version__))
import operator
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
# print('numpy: {}'.format(numpy.__version__))
# # matplotlib
# import matplotlib
# print('matplotlib: {}'.format(matplotlib.__version__))
# # # pandas
# # import pandas
# # print('pandas: {}'.format(pandas.__version__))
# # # scikit-learn
# import sklearn
# print('sklearn: {}'.format(sklearn.__version__))

# x = numpy.array([3, 1, 0])
# print(x,"\n",numpy.argsort(x))

# a = numpy.array([0, 1, 2])
# b = numpy.tile(a, 1)
# c= numpy.tile(a, 2)
# print(b,c)
# # x = numpy.array([1, 2, 3])
# print(x,"\n",numpy.argsort(x))
# x = np.array([1,2, 2, 4])
# y = x.shape[0]
# z= x.sum()
# print(z)
# from sklearn.neighbors.nearest_centroid import NearestCentroid
# import numpy as np
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# y = np.array([1, 1, 1, 2, 2, 2])
# clf = NearestCentroid()
# clf.fit(X, y)
# NearestCentroid(metric='euclidean', shrink_threshold=None)
# print(clf.predict([[-1.8, -1]]))
group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
labels = ['A','A','B','B']
dataSet = group
inX =[0,0]
dataSetSize = dataSet.shape[0]
diffMat = tile(inX, (dataSetSize,1)) - dataSet
sqDiffMat = diffMat**2
sqDistances = sqDiffMat.sum(axis=1)
distances = sqDistances**0.5
sortedDistIndicies = distances.argsort() 
k=3
dataSetSize = dataSet.shape[0]
diffMat = tile(inX, (dataSetSize,1)) - dataSet
sqDiffMat = diffMat**2
sqDistances = sqDiffMat.sum(axis=1)
distances = sqDistances**0.5
sortedDistIndicies = distances.argsort()     
classCount={}                   
for i in range(k):
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
print(sortedClassCount[0][0])