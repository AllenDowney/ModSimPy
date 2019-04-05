import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import timeit


iris = datasets.load_iris()


start = timeit.timeit()
# print(iris.data.shape,iris.target.shape)

X_train, X_test, y_train, y_test = train_test_split(
         iris.data, iris.target, test_size=0.4, random_state=0)

# print(len(X_train),len(X_test))

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

end = timeit.timeit()

print(start,clf.score(X_test, y_test))