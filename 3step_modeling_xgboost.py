#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[ ]:


get_ipython().run_line_magic('watermark', '')


# In[ ]:


import os
import pickle as pkl

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn import preprocessing


# In[ ]:


from xgboost import __version__ as xgb_version
from sklearn import __version__ as sklearn_version

print(f'xgb_version: {xgb_version}')
print(f'sklearn_version: {sklearn_version}')


# In[ ]:


get_ipython().run_line_magic('watermark', '--iversions')


# In[ ]:





# ## Reproducibility block

# In[ ]:


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

CB_RANDOMSEED = 309487
XGB_RANDOMSEED = 56


# In[ ]:





# In[ ]:


DIR_DATA   = os.path.join(os.getcwd(), 'data')
DIR_MODELS = os.path.join(os.getcwd(), 'models')
DIR_SUBM   = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_PART = os.path.join(os.getcwd(), 'subm', 'partial')


# In[ ]:





# ## Load data

# In[ ]:


x_train  = pd.read_csv(os.path.join(DIR_DATA, 'x_train.csv'), index_col= 0)
x_val    = pd.read_csv(os.path.join(DIR_DATA, 'x_val.csv'), index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'), index_col= 0)

with open(os.path.join(DIR_DATA, 'cat_columns.pkl'), 'rb') as pickle_file:
    cat_cols = pkl.load(pickle_file)
    
with open(os.path.join(DIR_DATA, 'num_columns.pkl'), 'rb') as pickle_file:
    num_cols = pkl.load(pickle_file)
    
with open(os.path.join(DIR_DATA, 'clmns.pkl'), 'rb') as pickle_file:
    clmns = pkl.load(pickle_file)


# In[ ]:


x_train.shape, x_val.shape, df_test.shape, len(cat_cols), len(num_cols)


# отделяем метки от данных

# In[ ]:


y_train = x_train[['views', 'depth', 'full_reads_percent']]
y_val   = x_val[['views', 'depth', 'full_reads_percent']]

x_train.drop(['views', 'depth', 'full_reads_percent'], axis = 1, inplace = True)
x_val.drop(  ['views', 'depth', 'full_reads_percent'], axis = 1, inplace = True)

x_train.shape, x_val.shape, y_train.shape, y_val.shape


# In[ ]:


#cat_cols + num_cols


# In[ ]:


cat_cols = []
num_cols = []

for el in clmns.keys():
    cat_cols.extend(clmns[el]['cat'])
    num_cols.extend(clmns[el]['num'])
    if len(clmns[el]['both']) != 0:
        print(clmns[el]['both'])


# In[ ]:


num_cols.extend(['hour', 'mounth', 'dow'])
#cat_cols.extend(['dow'])


# In[ ]:





# In[ ]:


def plot_importance(inp_model, imp_number = 30, imp_type = 'weight'):
    feature_important = inp_model.get_booster().get_score(importance_type=imp_type)
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    data.nlargest(imp_number, columns="score").plot(kind='barh', figsize = (30,16)) ## plot top 40 features


# In[ ]:





# ## views

# In[ ]:


xgb_model_views = XGBRegressor(n_estimators=1000, 
                               max_depth=7, 
                               eta=0.1, 
                               #subsample=0.7, 
                               colsample_bytree=0.8,
                               n_jobs = -1,
                               random_state = XGB_RANDOMSEED,
                              )

xgb_model_views.fit(x_train[num_cols], y_train['views'], 
                    early_stopping_rounds=5,
                    eval_set=[(x_val[num_cols], y_val['views'])], 
                    verbose=False
                   )


# In[ ]:





# In[ ]:


# Get predictions and metrics
preds_train_views = xgb_model_views.predict(x_train[num_cols])
preds_val_views   = xgb_model_views.predict(x_val[num_cols])

train_score_views = r2_score(y_train["views"], preds_train_views)
val_score_views   = r2_score(y_val["views"],   preds_val_views)

train_score_views, val_score_views

(0.7755792099701974, 0.8464698702781239) baseline
# In[ ]:





# In[ ]:


plot_importance(xgb_model_views, 30, 'weight')


# In[ ]:





# ## depth

# In[ ]:


xgb_model_depth = XGBRegressor(n_estimators=1000, 
                               max_depth=7, 
                               eta=0.1, 
                               #subsample=0.7, 
                               colsample_bytree=0.8,
                               n_jobs = -1,
                               random_state = XGB_RANDOMSEED,
                              )

xgb_model_depth.fit(x_train[num_cols], y_train['depth'], 
                    early_stopping_rounds=5,
                    eval_set=[(x_val[num_cols], y_val['depth'])], 
                    verbose=False
                   )


# In[ ]:


# Get predictions and metrics
preds_train_depth = xgb_model_depth.predict(x_train[num_cols])
preds_val_depth   = xgb_model_depth.predict(x_val[num_cols])

train_score_depth = r2_score(y_train["depth"], preds_train_depth)
val_score_depth   = r2_score(y_val["depth"],   preds_val_depth)

train_score_depth, val_score_depth

(0.9507037589614913, 0.7433207812829997) emb + lags
# In[ ]:





# In[ ]:


plot_importance(xgb_model_depth, 30, 'weight')


# In[ ]:





# ## full_reads_percent

# In[ ]:


#pd.DataFrame(preds_train_depth, columns = ['depth_pred'])
pred_scaler = preprocessing.StandardScaler()
tmp = pred_scaler.fit_transform(preds_train_depth.reshape(-1, 1))
pred_depth_train = pd.DataFrame(tmp, columns = ['depth_pred'])

pred_depth_val   = pd.DataFrame(pred_scaler.transform(preds_val_depth.reshape(-1, 1)), columns = ['depth_pred'])

print('before ', x_train.shape, x_val.shape, preds_train_depth.shape, preds_val_depth.shape)
x_train = pd.concat([x_train, pred_depth_train], axis = 1)
x_val   = pd.concat([x_val,   pred_depth_val],   axis = 1)
print('after  ', x_train.shape, x_val.shape)
# In[ ]:


xgb_model_frp = XGBRegressor(n_estimators=1000, 
                             max_depth=7, 
                             eta=0.1, 
                             #subsample=0.7, 
                             colsample_bytree=0.8,
                             n_jobs = -1,
                             random_state = XGB_RANDOMSEED,
                             )

xgb_model_frp.fit(x_train[num_cols], 
                  y_train['full_reads_percent'], 
                  early_stopping_rounds=5,
                  eval_set=[(x_val[num_cols], 
                             y_val['full_reads_percent'])
                           ], 
                  verbose=False
                 )


# In[ ]:


# Get predictions and metrics
preds_train_frp = xgb_model_frp.predict(x_train[num_cols])
preds_val_frp   = xgb_model_frp.predict(x_val[num_cols])

train_score_frp = r2_score(y_train["full_reads_percent"], preds_train_frp)
val_score_frp   = r2_score(y_val["full_reads_percent"],   preds_val_frp)

train_score_frp, val_score_frp

(0.8743266355474598, 0.37379638051659225) emb + lags + nauth + all_norm
# In[ ]:





# In[ ]:


plot_importance(xgb_model_frp, 30, 'weight')


# In[ ]:





# In[ ]:


score_train = 0.4 * train_score_views + 0.3 * train_score_depth + 0.3 * train_score_frp
score_val   = 0.4 * val_score_views   + 0.3 * val_score_depth   + 0.3 * val_score_frp

score_train, score_val

(0.46553643466563366, 0.48177658679349394)
# In[ ]:





# In[ ]:


NTRY = 7


# ## save models

# In[ ]:


xgb_model_views.save_model(os.path.join(DIR_MODELS, f'{NTRY}_xgb_views.json'), 
                          )

xgb_model_depth.save_model(os.path.join(DIR_MODELS, f'{NTRY}_xgb_depth.json'), 
                          )

xgb_model_frp.save_model(os.path.join(DIR_MODELS, f'{NTRY}_xgb_frp.json'), 
                        )


# In[ ]:





# ## make predict

# In[ ]:


pred_views = xgb_model_views.predict(df_test[num_cols])
pred_depth = xgb_model_depth.predict(df_test[num_cols])
pred_frp   = xgb_model_frp.predict(  df_test[num_cols])


# In[ ]:


subm = pd.DataFrame()
subm['document_id'] = df_test.document_id

subm['views'] = pred_views
subm['depth'] = pred_depth
subm['full_reads_percent'] = pred_frp


# In[ ]:


subm.head()


# In[ ]:


subm.to_csv(os.path.join(DIR_SUBM, f'{NTRY}_xgb_lags_emb.csv'), index = False)


# In[ ]:




