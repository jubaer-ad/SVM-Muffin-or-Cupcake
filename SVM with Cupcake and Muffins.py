#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Alanysis tools
import pandas as pd
import numpy as np
from sklearn import svm

# Visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

#Allow charts to appear
get_ipython().run_line_magic('matplotlib', 'inline')

#pickle package
import pickle


# In[5]:


#read data
recipes = pd.read_csv(r'G:\ML\Dataset\muffin-cupcake-master\recipes_muffins_cupcakes.csv')
recipes


# In[6]:


# plot two ingredients
sns.lmplot('Flour', 'Sugar', data = recipes, hue ='Type', 
           palette = 'Set2', fit_reg = False, scatter_kws = {"s": 50})


# In[7]:


# specify inputs for the model
ingredients = recipes[['Flour', 'Sugar']].as_matrix()
type_label = np.where(recipes['Type'] == 'Muffin', 0, 1)

# feature names
recipe_features = recipes.columns.values[1:].tolist()
recipe_features


# In[8]:


# fit the svm model
model = svm.SVC(kernel = 'linear')
model.fit(ingredients, type_label)


# In[18]:


# get the hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (model.intercept_[0] / w[1])

#plotting parallels
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


# In[19]:


# plot hyperplane
sns.lmplot('Flour', 'Sugar', data = recipes, hue = 'Type', palette = 'Set2',
          fit_reg = False, scatter_kws = {"s": 70})
plt.plot(xx, yy, linewidth = 2, color = 'black')


# In[20]:


# margin
sns.lmplot('Flour', 'Sugar', data = recipes, hue = 'Type', palette = 'Set2',
          fit_reg = False, scatter_kws = {"s": 70})
plt.plot(xx, yy, linewidth = 2, color = 'black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
           s = 80, facecolor = 'none')


# In[21]:


# function to guess muffin or cupcake
def muffin_or_cupcake(flour, sugar):
    if(model.predict([[flour, sugar]])) == 0:
        print('You\'re muffin.')
    else:
        print('You\'re cupcake.')


# In[23]:


# test
muffin_or_cupcake(10, 20)


# In[24]:


# visual plotting
sns.lmplot('Flour', 'Sugar', data = recipes, hue = 'Type', palette = 'Set2',
          fit_reg = False, scatter_kws = {"s": 70})
plt.plot(xx, yy, linewidth = 2, color = 'black')
plt.plot(50, 20, 'yo', markersize = '9')


# In[29]:


# test again
muffin_or_cupcake(5, 2)


# In[30]:


# visual plotting again
sns.lmplot('Flour', 'Sugar', data = recipes, hue = 'Type', palette = 'Set2',
          fit_reg = False, scatter_kws = {"s": 70})
plt.plot(xx, yy, linewidth = 2, color = 'black')
plt.plot(50, 20, 'yo', markersize = '9')


# In[ ]:




