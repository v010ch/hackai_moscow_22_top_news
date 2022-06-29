#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[10]:


import os
#import pickle as pkl

#import numpy as np
import pandas as pd


# In[ ]:





# In[11]:


DIR_SUBM   = os.path.join(os.getcwd(), 'subm')


# In[12]:


subm_views = pd.read_csv(os.path.join(DIR_SUBM, '1_xgb_baseline_test.csv'),     usecols=['document_id', 'views'])
subm_depth = pd.read_csv(os.path.join(DIR_SUBM, '3_lgb_ttl_emb_depth_frp.csv'), usecols=['document_id', 'depth'])
subm_frp   = pd.read_csv(os.path.join(DIR_SUBM, '3_lgb_ttl_emb_depth_frp.csv'), usecols=['document_id', 'full_reads_percent'])


# In[16]:


subm = subm_views.merge(subm_depth, on='document_id', validate='one_to_one')
#subm = subm.merge(subm_depth, on='document_id', validate='one_to_one')
subm = subm.merge(subm_frp, on='document_id', validate='one_to_one')


# In[17]:


subm


# In[ ]:





# In[19]:


subm.to_csv(os.path.join(DIR_SUBM, '3__concatenate.csv'), index = False)


# In[ ]:




