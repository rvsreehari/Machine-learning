#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


X_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
Y_train = np.array([2, 3, 4, 5, 7, 8, 8, 9, 10, 10])


# In[3]:


class UnivariateLinearRegression:

  def __init__(self):
    self.theta_0 = None
    self.theta_1 = None

  def hypothesis(self,x):
    return self.theta_0 + self.theta_1*x

  def grad_theta_0 (self,x,y):
    ypred = self.hypothesis(x)
    return (ypred-y)

  def grad_theta_1 (self,x,y):
    ypred = self.hypothesis(x)
    return (ypred-y)*x

  def fit(self,X,Y,epochs=1,learning_rate=.01):

    self.theta_0 = 0.0
    self.theta_1 = 0.0
    m = X.shape[0]
    for i in range(epochs):
      dtheta_0 = 0.0
      dtheta_1 = 0.0
      for x,y in zip(X,Y): 
        dtheta_0 = dtheta_0 + self.grad_theta_0(x,y)
        dtheta_1 = dtheta_1 + self.grad_theta_1(x,y)
      self.theta_0 = self.theta_0 - learning_rate * dtheta_0 / m
      self.theta_1 = self.theta_1 - learning_rate * dtheta_1 / m
      #self.theta = self.theta - (alpha * (1/m) * np.dot(X.T, error))- Multi
       # Normalize our features
        #X = (X - X.mean()) / X.std()

  def predict(self,X):
    y_pred = []
    for x in X:
      y_pred.append(self.hypothesis(x))
    return np.array(y_pred)


# In[4]:


def predict(self,X):
    y_pred = []
    for x in X:
      y_pred.append(self.hypothesis(x))
    return np.array(y_pred)


# In[5]:


model = UnivariateLinearRegression()


# In[6]:


model.fit(X_train,Y_train,epochs=40000,learning_rate=.001)


# In[7]:


X_test = np.array([10, 11, 12, 13, 14, 15])
Y_test = np.array([12, 13, 14, 15, 16, 17])


# In[8]:


Y_pred = model.predict(X_test)


# In[9]:


from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[10]:


error = mean_squared_error(Y_pred,Y_test)
print(error)


# In[11]:


plt.scatter(X_test, Y_test,  color='black')
plt.plot(X_test, Y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()

