#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import os
#import pickle as pkl

#import numpy as np
import pandas as pd


# In[ ]:





# In[ ]:


DIR_SUBM   = os.path.join(os.getcwd(), 'subm')


# In[ ]:


subm_views = pd.read_csv(os.path.join(DIR_SUBM, '1_xgb_baseline_test.csv'),     usecols=['document_id', 'views'])
subm_depth = pd.read_csv(os.path.join(DIR_SUBM, '7_xgb_lags_emb.csv'),          usecols=['document_id', 'depth'])
subm_frp   = pd.read_csv(os.path.join(DIR_SUBM, '7_xgb_lags_emb.csv'), usecols=['document_id', 'full_reads_percent'])
print(subm_views.shape, subm_depth.shape, subm_frp.shape)


# In[ ]:


subm = subm_views.merge(subm_depth, on='document_id', validate='one_to_one')
#subm = subm.merge(subm_depth, on='document_id', validate='one_to_one')
subm = subm.merge(subm_frp, on='document_id', validate='one_to_one')


# In[ ]:





# In[ ]:


NTYPE = 7


# In[ ]:


subm.to_csv(os.path.join(DIR_SUBM, f'{NTYPE}_concatenate.csv'), index = False)


# In[ ]:




