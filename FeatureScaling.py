#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[12]:


R = np.random.random([100,1]) + 3
print(R)


# In[3]:


plt.plot(R)


# In[4]:


np.mean(R)


# In[5]:


np.std(R)


# In[6]:


scalar = StandardScaler()


# In[7]:


scalar.fit(R)


# In[8]:


scalar.mean_


# In[9]:


RScaled = scalar.transform(R)


# In[10]:


plt.plot(RScaled)


# In[11]:


np.mean(RScaled)

