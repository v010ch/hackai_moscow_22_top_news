#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import pandas as pd
import pickle as pkl

from sklearn.model_selection import cross_val_score


# In[2]:


import torch
from torch import nn


# In[ ]:





# In[3]:


#dir(nn)


# In[ ]:





# In[ ]:





# In[4]:


DIR_DATA   = os.path.join(os.getcwd(), 'data')
DIR_MODELS = os.path.join(os.getcwd(), 'models')
DIR_SUBM   = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_PART = os.path.join(os.getcwd(), 'subm', 'partial')


# In[ ]:





# In[5]:


NTRY = 11
NAME = f'{NTRY}_nn_pca64_ttls_sber_lags_parse'


# In[ ]:





# ## Загружаем данные

# In[6]:


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'))#, index_col= 0)
x_train  = pd.read_csv(os.path.join(DIR_DATA, 'x_train.csv'))#, index_col= 0)
x_val    = pd.read_csv(os.path.join(DIR_DATA, 'x_val.csv'))#, index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'))#, index_col= 0)

with open(os.path.join(DIR_DATA, 'cat_columns.pkl'), 'rb') as pickle_file:
    cat_cols = pkl.load(pickle_file)
    
with open(os.path.join(DIR_DATA, 'num_columns.pkl'), 'rb') as pickle_file:
    num_cols = pkl.load(pickle_file)

with open(os.path.join(DIR_DATA, 'clmns.pkl'), 'rb') as pickle_file:
    clmns = pkl.load(pickle_file)


# In[7]:


cat_cols = []
num_cols = []

for el in clmns.keys():
    cat_cols.extend(clmns[el]['cat'])
    num_cols.extend(clmns[el]['num'])
    if len(clmns[el]['both']) != 0:
        #print(clmns[el]['both'])
        cat_cols.extend(clmns[el]['both'])


# In[ ]:





# In[ ]:





# In[8]:


mult = 1.5


# In[9]:


model = nn.Sequential()
model.add_module('inp_layer', nn.Linear(len(num_cols), int(len(num_cols) * mult)))
model.add_module('hid_layer1', nn.Linear(int(len(num_cols) * mult),  int(len(num_cols) * 0.7)))
model.add_modile('drop1', nn.Droupout(0.2))
model.add_module('hid_layer2', nn.Linear(int(len(num_cols) * 0.7)), 100)
model.add_module('out_layer', nn.Linear(100, 1))


# In[ ]:





# In[ ]:





# In[ ]:


class rbk_model:
    

