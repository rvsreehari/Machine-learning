#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# In[18]:


X = np.linspace(0,.5,100)
X.shape


# In[19]:


y_actual =  np.cos(2*np.pi*X)**2
y_actual.shape


# In[20]:


plt.plot(X,y_actual)


# In[21]:


noise = np.random.normal(0, 0.1, 100)
print(noise)


# In[22]:


y = y_actual + noise
plt.plot(X,y_actual)
plt.scatter(X,y,color="red")


# In[23]:


model = linear_model.LinearRegression(fit_intercept="True")


# In[24]:


# over fitting


# In[25]:


X=X.reshape(-1,1)
model.fit(X,y)


# In[26]:


y_pred = model.predict(X)


# In[27]:


plt.plot(X,y_actual)
plt.scatter(X,y,color="red")
plt.plot(X,y_pred,color='green')


# In[28]:


deg = 2
scalar = StandardScaler()
polynomial_features = PolynomialFeatures(degree=deg,include_bias="True")
linear = linear_model.LinearRegression(fit_intercept="True")
pipeline = Pipeline([("polynomial",polynomial_features),("scaling",scalar),("linear",linear)])


# In[29]:


pipeline.fit(X,y)


# In[30]:


y_pred = pipeline.predict(X)


# In[31]:


plt.plot(X,y_actual)
plt.scatter(X,y,color="red")
plt.plot(X,y_pred,color='green')


# In[32]:


#over fit
deg = 10000
polynomial_features = PolynomialFeatures(degree=deg,include_bias="True")
linear = linear_model.LinearRegression(fit_intercept="True")
pipeline = Pipeline([("polynomial",polynomial_features),("linear",linear)])
pipeline.fit(X,y)
y_pred = pipeline.predict(X)
plt.plot(X,y_actual)
plt.scatter(X,y,color="red")
plt.plot(X,y_pred,color='green')


# In[ ]:




