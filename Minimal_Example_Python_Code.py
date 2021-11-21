#!/usr/bin/env python
# coding: utf-8

# ## Import the relevant libraries

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ### Generate random input data to train on

# In[4]:


# First, we should declare a variable containing the size of the training set we want to generate.
observations = 1000

xs = np.random.uniform(low=-10, high=10, size=(observations,1))
zs = np.random.uniform(-10, 10, (observations,1))

# Combine the two dimensions of the input into one input matrix. 
# This is the X matrix from the linear model y = x*w + b.
inputs = np.column_stack((xs,zs))

# Check the shape of inputs:
print (inputs.shape)


# ### Generate the targets we will aim at

# In[5]:


# f(x,z) = 2x - 3z + 5 + noise

# noise: random[-1,1]:
noise = np.random.uniform(-1, 1, (observations,1))

# Supervised parameters:
targets = 2*xs - 3*zs + 5 + noise

print (targets.shape)


# ### Initialize variables (parameters)
# 

# In[6]:


init_range = 0.1
weights = np.random.uniform(low=-init_range, high=init_range, size=(2, 1))
biases = np.random.uniform(low=-init_range, high=init_range, size=1)
print (weights)
print (biases)


# ### Set a learning rate
# 

# In[7]:


# 0.02 is going to work quite well for our example. You can play around with it.

learning_rate = 0.02


# ### Train the model
# 

# In[8]:


for i in range (100):
    # This is the linear model: y = xw + b equation
    outputs = np.dot(inputs,weights) + biases
    
    # The deltas are the differences between the outputs and the targets
    deltas = outputs - targets
    
    # Our Objective Function: L2-norm Loss
    # divided by 2 and observations is just for rescaling purposes, it doesn't change the ubderlying logic of the loss function
    # to make it independent from the number of samples:
    loss = np.sum(deltas ** 2) / 2 / observations
    print (loss)
    
    deltas_scaled = deltas / observations
    
    # Gradient Descent update rules:
    weights = weights - learning_rate * np.dot(inputs.T,deltas_scaled)
    biases = biases - learning_rate * np.sum(deltas_scaled)


# ### Print weights and biases and see if we have worked correctly.

# In[9]:


print (weights, biases)


# In[ ]:




