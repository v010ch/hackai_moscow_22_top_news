#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[2]:


get_ipython().run_line_magic('watermark', '')


# In[3]:


import os
import pickle as pkl

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
import lightgbm as lgb


# In[4]:


#from catboost import __version__ as cb_version
from sklearn import __version__ as sklearn_version

#print(f'cb_version: {cb_version}')
print(f'sklearn_version: {sklearn_version}')


# In[5]:


get_ipython().run_line_magic('watermark', '--iversions')


# In[ ]:





# In[ ]:





# ## Reproducibility block

# In[6]:


# seed the RNG for all devices (both CPU and CUDA)
#torch.manual_seed(1984)

#Disabling the benchmarking feature causes cuDNN to deterministically select an algorithm, 
#possibly at the cost of reduced performance.
#torch.backends.cudnn.benchmark = False

# for custom operators,
import random
random.seed(5986721)

# 
np.random.seed(62185)

#sklearn take seed from a line abowe

CB_RANDOMSEED  = 309487
XGB_RANDOMSEED = 56
LGB_RANDOMSEED = 874256


# In[ ]:





# In[7]:


DIR_DATA   = os.path.join(os.getcwd(), 'data')
DIR_MODELS = os.path.join(os.getcwd(), 'models')
DIR_SUBM   = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_PART = os.path.join(os.getcwd(), 'subm', 'partial')


# In[ ]:





# ## Load data

# In[8]:


x_train  = pd.read_csv(os.path.join(DIR_DATA, 'x_train.csv'), index_col= 0)
x_val    = pd.read_csv(os.path.join(DIR_DATA, 'x_val.csv'), index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'), index_col= 0)

with open(os.path.join(DIR_DATA, 'cat_columns.pkl'), 'rb') as pickle_file:
    cat_cols = pkl.load(pickle_file)
    
with open(os.path.join(DIR_DATA, 'num_columns.pkl'), 'rb') as pickle_file:
    num_cols = pkl.load(pickle_file)


# In[9]:


x_train.shape, x_val.shape, df_test.shape, len(cat_cols), len(num_cols)


# отделяем метки от данных

# In[10]:


y_train = x_train[['views', 'depth', 'full_reads_percent']]
y_val   = x_val[['views', 'depth', 'full_reads_percent']]

x_train.drop(['views', 'depth', 'full_reads_percent'], axis = 1, inplace = True)
x_val.drop(  ['views', 'depth', 'full_reads_percent'], axis = 1, inplace = True)

x_train.shape, x_val.shape, y_train.shape, y_val.shape


# In[11]:


cat_cols + num_cols


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


#lgb_train = lgb.Dataset(x_train, y_train)
#lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

#views
train_ds_views = lgb.Dataset(x_train[cat_cols + num_cols],
                             y_train[['views']],
                             #feature_name = [cat_cols + num_cols]
                            )
val_ds_views = lgb.Dataset(x_val[cat_cols + num_cols],
                             y_val[['views']],
                             #feature_name = [cat_cols + num_cols]
                            )


#depth
train_ds_depth = lgb.Dataset(x_train[cat_cols + num_cols],
                             y_train[['depth']],
                             #feature_name = [cat_cols + num_cols]
                            )
val_ds_depth = lgb.Dataset(x_val[cat_cols + num_cols],
                             y_val[['depth']],
                             #feature_name = [cat_cols + num_cols]
                            )


#full_reads_percent
train_ds_frp = lgb.Dataset(x_train[cat_cols + num_cols],
                             y_train[['full_reads_percent']],
                             #feature_name = [cat_cols + num_cols]
                            )
val_ds_frp = lgb.Dataset(x_val[cat_cols + num_cols],
                             y_val[['full_reads_percent']],
                             #feature_name = [cat_cols + num_cols]
                            )
#train_ds_frp
#val_ds_frp


# In[ ]:





# In[ ]:




gbdt, traditional Gradient Boosting Decision Tree, aliases: gbrt
rf, Random Forest, aliases: random_forest
dart, Dropouts meet Multiple Additive Regression Trees
goss, Gradient-based One-Side Sampling
# ## views

# In[13]:


# defining parameters 
params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    'learning_rates': 0.05,
    'metric': {'l2','l1'},
    'verbose': -1,
    'random_seed': LGB_RANDOMSEED,
}


# In[14]:


# fitting the model
lgb_model_views = lgb.train(params,
                            train_set=train_ds_views,
                            valid_sets=val_ds_views,
                            early_stopping_rounds=30,
                            verbose_eval = False,
                           )


# In[15]:


# Get predictions and metrics
preds_train_views = lgb_model_views.predict(x_train[cat_cols + num_cols])
preds_val_views   = lgb_model_views.predict(x_val[cat_cols + num_cols])

train_score_views = r2_score(y_train["views"], preds_train_views)
val_score_views   = r2_score(y_val["views"],   preds_val_views)

train_score_views, val_score_views

(0.598891443106734, 0.6282297125590693)
# In[ ]:





# In[ ]:





# ## depth

# In[16]:


# defining parameters 
params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    'learnig_rates': 0.05,
    'metric': {'l2','l1'},
    'verbose': -1,
    'random_seed': LGB_RANDOMSEED,
}


# In[17]:


# fitting the model
lgb_model_depth = lgb.train(params,
                            train_set=train_ds_depth,
                            valid_sets=val_ds_depth,
                            early_stopping_rounds=30,
                            verbose_eval = False,
                           )


# In[18]:


# Get predictions and metrics
preds_train_depth = lgb_model_depth.predict(x_train[cat_cols + num_cols])
preds_val_depth   = lgb_model_depth.predict(x_val[cat_cols + num_cols])

train_score_depth = r2_score(y_train["depth"], preds_train_depth)
val_score_depth   = r2_score(y_val["depth"],   preds_val_depth)

train_score_depth, val_score_depth

(0.7268344081845861, 0.6434700700442351)
# In[ ]:





# ## full_reads_percent

# In[19]:


# defining parameters 
params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    'learnig_rates': 0.05,
    'metric': {'l2','l1'},
    'verbose': -1,
    'random_seed': LGB_RANDOMSEED,
}


# In[20]:


# fitting the model
lgb_model_frp = lgb.train(params,
                            train_set=train_ds_frp,
                            valid_sets=val_ds_frp,
                            early_stopping_rounds=30,
                            verbose_eval = False,
                           )


# In[21]:


# Get predictions and metrics
preds_train_frp = lgb_model_frp.predict(x_train[cat_cols + num_cols])
preds_val_frp  = lgb_model_frp.predict(x_val[cat_cols + num_cols])

train_score_frp = r2_score(y_train["full_reads_percent"], preds_train_frp)
val_score_frp  = r2_score(y_val["full_reads_percent"],   preds_val_frp)

train_score_frp, val_score_frp

(0.30008084990246797, 0.2593834027777585)
# In[ ]:





# In[22]:


score_train = 0.4 * train_score_views + 0.3 * train_score_depth + 0.3 * train_score_frp
score_val  = 0.4 * val_score_views  + 0.3 * val_score_depth  + 0.3 * val_score_frp

score_train, score_val

(0.5476311546688098, 0.5221479268702258)
# In[ ]:





# ## save models

# In[23]:


lgb_model_views.save_model(os.path.join(DIR_MODELS, 'lgm_views.txt'), num_iteration = lgb_model_views.best_iteration)
lgb_model_depth.save_model(os.path.join(DIR_MODELS, 'lgm_depth.txt'), num_iteration = lgb_model_depth.best_iteration)
lgb_model_frp.save_model(  os.path.join(DIR_MODELS, 'lgm_frp.txt'),   num_iteration = lgb_model_frp.best_iteration)


# In[ ]:





# ## make predict

# In[24]:


pred_views = lgb_model_views.predict(df_test[cat_cols + num_cols])
pred_depth = lgb_model_depth.predict(df_test[cat_cols + num_cols])
pred_frp   = lgb_model_frp.predict(  df_test[cat_cols + num_cols])


# In[25]:


subm = pd.DataFrame()
subm['document_id'] = df_test.index

subm['views'] = pred_views
subm['depth'] = pred_depth
subm['full_reads_percent'] = pred_frp


# In[26]:


subm.head()


# In[27]:


subm.to_csv(os.path.join(DIR_SUBM, '1_lgb_baseline_test.csv'), index = False)


# In[ ]:




