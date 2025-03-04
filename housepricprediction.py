#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import statsmodels.api as sm


# In[5]:


california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)


# In[9]:


df["PRICE"] = california.target  


# In[11]:


df.head()


# In[13]:


df.isnull().sum()


# In[15]:


df.describe()


# In[17]:


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[21]:


X = df["MedInc"]
Y = df["PRICE"]


# In[23]:


X = sm.add_constant(X)


# In[25]:


model = sm.OLS(Y, X).fit()


# In[27]:


print(model.summary())


# In[29]:


sns.regplot(x=df["MedInc"], y=df["PRICE"], line_kws={"color": "red"})
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("Linear Regression: Income vs Price")
plt.show()


# In[ ]:




