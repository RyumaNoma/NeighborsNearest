from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt

class NeighborsNearest:
  def __init__(self):
    self.X = []
    self.y = []
    
  def calc(self, dest, test_point):
    sum = 0.0
    
    for f1, f2 in zip(dest, test_point):
      sum += np.sqrt((f1 - f2) * (f1 - f2))
      
    return sum
    
  def fit(self, X, y):
    self.X = X
    self.y = y
    self.train_size = len(self.X)
    
    return self
    
  def predict(self, feature):
    min_No = self.y[0]
    min_score = self.calc(feature, self.X[0])
    
    for i in np.arange(self.train_size):
      now = self.calc(feature, self.X[i])
      if min_score > now:
        min_score = now
        min_No = self.y[i]
        
    return min_No
  
  def score(self, X_test, y_test):
    count = 0
    
    for data, target in zip(X_test, y_test):
      if target == self.predict(data):
        count += 1
        
    return count / len(y_test)

# アヤメ分類問題を用いて正しく動作しているか確認する
# アヤメデータセットの準備
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

# テストデータ[5.8, 2.8, 5.1, 2.4]を予想してみる（答えは[2]）

# scikit-learnに実装されているK近傍法を利用する
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print(knn.predict([[5.8, 2.8, 5.1, 2.4]])) #=> [2]

# X_train, y_trainを用いて学習する
k = NeighborsNearest()
k.fit(X_train, y_train)
print(k.predict([5.8, 2.8, 5.1, 2.4])) #=>　2

# X_test, y_testを予想する
print(knn.score(X_test, y_test)) #=> 0.9736842105263158
print(k.score(X_test, y_test)) #=> 0.9736842105263158