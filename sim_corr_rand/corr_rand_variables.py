#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from scipy.linalg import cholesky
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skewstudent_master.skewstudent.skewstudent import SkewStudent


# In[45]:


nscen = 100000
# x1 = np.random.normal(size=nscen)
skewt = SkewStudent(eta=8, lam=-0.5)
x1 = skewt.rvs(nscen)
x2 = np.random.normal(size=nscen)
marginals = np.stack([x1, x2], axis=0)
display(marginals, np.cov(marginals), np.corrcoef(marginals))


# In[46]:


rho = -0.4
covar_mat = np.array([[1, rho], [rho, 1]])
chol = cholesky(covar_mat, lower=True)
corr_rands = chol @ marginals
display(corr_rands, np.cov(corr_rands), np.corrcoef(corr_rands))


# In[48]:


# sns.set(); sns.set_context('notebook')
fig, axes = plt.subplots(1, 2, figsize=(12,6))
sns.histplot([marginals[0], marginals[1]], ax=axes[0], kde=True)
axes[0].legend(['Marginal 1', 'Marginal 2'])
sns.scatterplot(corr_rands[0], corr_rands[1], ax=axes[1])
axes[1].set(xlabel='Marginal 1', ylabel='Marginal 2', title=f'Correlation = {rho}')
# axes[1].scatter(corr_rands[0], corr_rands[1])


# In[ ]:




