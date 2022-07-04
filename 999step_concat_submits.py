#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import os
#import pickle as pkl

#import numpy as np
import pandas as pd


# In[ ]:





# In[2]:


DIR_SUBM   = os.path.join(os.getcwd(), 'subm')


# In[3]:


subm_views = pd.read_csv(os.path.join(DIR_SUBM, '1_xgb_baseline_test.csv'),     usecols=['document_id', 'views'])
subm_depth = pd.read_csv(os.path.join(DIR_SUBM, '5_lgb_ttl_emb_depth_frp.csv'), usecols=['document_id', 'depth'])
subm_frp   = pd.read_csv(os.path.join(DIR_SUBM, '5_lgb_ttl_emb_depth_frp.csv'), usecols=['document_id', 'full_reads_percent'])
print(subm_views.shape, subm_depth.shape, subm_frp.shape)


# In[4]:


subm = subm_views.merge(subm_depth, on='document_id', validate='one_to_one')
#subm = subm.merge(subm_depth, on='document_id', validate='one_to_one')
subm = subm.merge(subm_frp, on='document_id', validate='one_to_one')


# In[ ]:





# In[5]:


NTYPE = 5


# In[6]:


subm.to_csv(os.path.join(DIR_SUBM, f'{NTYPE}_concatenate.csv'), index = False)


# In[ ]:




