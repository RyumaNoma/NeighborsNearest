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
