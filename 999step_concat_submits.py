#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import os
#import pickle as pkl

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from xgboost import XGBRegressor
import xgboost as xgb

from sklearn import preprocessing
from sklearn.metrics import r2_score
from typing import Tuple


# In[ ]:





# In[3]:


DIR_DATA   = os.path.join(os.getcwd(), 'data')
DIR_MODELS = os.path.join(os.getcwd(), 'models')
DIR_SUBM   = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_PART = os.path.join(os.getcwd(), 'subm', 'partial')


# In[4]:


NTRY = 25
#NAME = f'{NTRY}_cb_pca64_sber_lags_parse_bord_nose'
#NAME_CB  = f'{NTRY}_cb_pca64_sber_bord_nose_iter_2mod'
#NAME_XGB = f'{NTRY}_xgb_pca64_sber_bord_nose_iter_2mod'
#NAME_LGB = f'{NTRY}_lgb_pca64_sber_bord_nose_iter_2mod'
#xgb_pca64_sber_lags_parse_bord_nose_val_part


# In[5]:


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





# # Конкатенация

# In[6]:


#subm_views = pd.read_csv(os.path.join(DIR_SUBM, f'1_xgb_baseline_test.csv'),  usecols=['document_id', 'views'])
subm_views = pd.read_csv(os.path.join(DIR_SUBM, f'{NTRY}_lgb_pca64_sber_bord_nose_iter_poly.csv'),  usecols=['document_id', 'views'])
subm_depth = pd.read_csv(os.path.join(DIR_SUBM, f'{NTRY}_lgb_pca64_sber_bord_nose_iter_poly.csv'), usecols=['document_id', 'depth'])
subm_frp   = pd.read_csv(os.path.join(DIR_SUBM, f'{NTRY}_lgb_pca64_sber_bord_nose_iter_poly.csv'), usecols=['document_id', 'full_reads_percent'])

#ubm_depth = pd.read_csv(os.path.join(DIR_SUBM, f'{NTRY}_lgb_pca64_sber_lags_parse_bord_nose_irq.csv'), usecols=['document_id', 'depth'])
#ubm_frp   = pd.read_csv(os.path.join(DIR_SUBM, f'14_lgb_pca64_sber_lags_parse_bord_nose.csv'),   usecols=['document_id', 'full_reads_percent'])

print(subm_views.shape, subm_depth.shape, subm_frp.shape)


# In[7]:


subm = subm_views.merge(subm_depth, on='document_id', validate='one_to_one')
#subm = subm.merge(subm_depth, on='document_id', validate='one_to_one')
subm = subm.merge(subm_frp, on='document_id', validate='one_to_one')


# In[8]:


subm.to_csv(os.path.join(DIR_SUBM, f'{NTRY}_concatenate.csv'), index = False)


# In[ ]:





# In[ ]:





# # Ансамбли

# In[48]:


def plot_importance(inp_model, imp_number = 30, imp_type = 'weight'):
    feature_important = inp_model.get_booster().get_score(importance_type=imp_type)
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    data.nlargest(imp_number, columns="score").plot(kind='barh', figsize = (30,16)) ## plot top 40 features


# In[8]:


train  = pd.read_csv(os.path.join(DIR_DATA, 'x_train.csv'), usecols = ['document_id', 'views', 'depth', 'full_reads_percent'])
val  = pd.read_csv(os.path.join(DIR_DATA, 'x_val.csv'), usecols = ['document_id', 'views', 'depth', 'full_reads_percent'])

cb_train = pd.read_csv(os.path.join(DIR_SUBM_PART, f'{NAME_CB}_train_part.csv'))
cb_val = pd.read_csv(os.path.join(DIR_SUBM_PART, f'{NAME_CB}_val_part.csv'))

xgb_train = pd.read_csv(os.path.join(DIR_SUBM_PART, f'{NAME_XGB}_train_part.csv'))
xgb_val = pd.read_csv(os.path.join(DIR_SUBM_PART, f'{NAME_XGB}_val_part.csv'))

lgb_train = pd.read_csv(os.path.join(DIR_SUBM_PART, f'{NAME_LGB}_train_part.csv'))
lgb_val = pd.read_csv(os.path.join(DIR_SUBM_PART, f'{NAME_LGB}_val_part.csv'))


# In[9]:


print('before ', cb_train.shape, lgb_train.shape, xgb_train.shape)
print('before ', cb_val.shape, lgb_val.shape, xgb_val.shape)
train = pd.concat([train.reset_index(drop = True),
                   cb_train.reset_index(drop = True),
                   xgb_train.reset_index(drop = True),
                   lgb_train.reset_index(drop = True),
                  ], ignore_index = True, axis = 1)

val = pd.concat([val.reset_index(drop = True),
                 cb_val.reset_index(drop = True),
                 xgb_val.reset_index(drop = True),
                 lgb_val.reset_index(drop = True),
                ], ignore_index = True, axis = 1)

cols = ['document_id', 'views', 'depth', 'full_reads_percent',
         'document_id_cb', 'views_pred_cb', 'depth_pred_cb', 'frp_pred_cb',
         'document_id_xgb', 'views_pred_xgb', 'depth_pred_xgb', 'frp_pred_xgb',
         'document_id_lgb', 'views_pred_lgb', 'depth_pred_lgb', 'frp_pred_lgb',
        ]
train.columns = cols
val.columns = cols

if sum(train.document_id == train.document_id_xgb) != train.shape[0] or    sum(train.document_id == train.document_id_lgb) != train.shape[0]:
    print('wtf train')
    
if sum(val.document_id == val.document_id_xgb) != val.shape[0] or    sum(val.document_id == val.document_id_lgb) != val.shape[0]:
    print('wtf val')

train.drop(['document_id_cb', 'document_id_xgb', 'document_id_lgb'], axis = 1, inplace = True)
val.drop(['document_id_cb', 'document_id_xgb', 'document_id_lgb'], axis = 1, inplace = True)
    
    
print('after ', train.shape)
print('after ', val.shape)


# In[11]:


val.columns[4:]


# In[ ]:





# # нормализуем

# In[56]:


scaler = preprocessing.StandardScaler()  #Standardize features by removing the mean and scaling to unit variance.

scaler.fit(train[train.columns[4:]])

train[train.columns[4:]] = scaler.transform(train[train.columns[4:]])
val[train.columns[4:]]  = scaler.transform(val[train.columns[4:]])


# In[74]:


#!!!!!!!!! нормализовать предикты
#!!!!!!!!! сделать предикты


# In[ ]:





# In[ ]:





# In[57]:


#def r2(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
def r2(y_pred: np.ndarray, y_true: xgb.DMatrix) -> Tuple[str, float]:
    
    #preds = np.array(y_pred[0])
    #print(type(y_true))
    #print(type(y_pred)) # np.array
    
    return 'r2', r2_score(y_true.get_label(), y_pred)


# # views

# In[58]:


cv_ntrees = 100

xgb_params_views = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    #'learning_rate': 0.05,
    'eta': 0.3,
    'max_depth': 15, 
    #'num_boost_round': 10000, 
    #'early_stopping_rounds': 100,
}
dtrain = xgb.DMatrix(train[train.columns[4:]], label=train[['views']])
dval   = xgb.DMatrix(val[train.columns[4:]], label=val[['views']])


# In[59]:


get_ipython().run_cell_magic('time', '', "score = xgb.cv(xgb_params_views, dtrain, cv_ntrees, nfold=5, #early_stopping_rounds=1000,\n        metrics={'rmse'},\n        custom_metric = r2,\n       #callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]\n      )\nscore.tail()")


# In[60]:


score[score['train-r2-mean'] == score['train-r2-mean'].max()][:1]


# In[61]:


score[score['test-r2-mean'] == score['test-r2-mean'].max()][:1]


# # сейчас учу только на тесте

# In[62]:


xgb_model_views = XGBRegressor(n_estimators=45, 
                               max_depth=15, 
                               eta=0.3, 
                               #subsample=0.7, 
                               #colsample_bytree=0.8,
                               n_jobs = -1,
                               random_state = XGB_RANDOMSEED,
                              )

xgb_model_views.fit(train[train.columns[4:]], train[['views']],
                    early_stopping_rounds=5,
                    eval_set=[(val[val.columns[4:]], val[['views']])], 
                    verbose=False
                   )

# Get predictions and metrics
preds_train_views = xgb_model_views.predict(train[train.columns[4:]])
preds_val_views   = xgb_model_views.predict(val[train.columns[4:]])

train_score_views = r2_score(train["views"], preds_train_views)
val_score_views   = r2_score(val["views"],   preds_val_views)

train_score_views, val_score_views


# In[63]:


plot_importance(xgb_model_views, 30, 'weight')


# In[ ]:





# # depth

# In[64]:


#cv_ntrees = 100

xgb_params_depth = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    #'learning_rate': 0.05,
    'eta': 0.3,
    'max_depth': 15, 
    #'num_boost_round': 10000, 
    #'early_stopping_rounds': 100,
}
dtrain = xgb.DMatrix(train[train.columns[4:]], label=train[['depth']])


# In[65]:


get_ipython().run_cell_magic('time', '', "score = xgb.cv(xgb_params_depth, dtrain, cv_ntrees, nfold=5, #early_stopping_rounds=1000,\n        metrics={'rmse'},\n        custom_metric = r2,\n       #callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]\n      )\nscore.tail()")


# In[66]:


score[score['train-r2-mean'] == score['train-r2-mean'].max()][:1]


# In[67]:


score[score['test-r2-mean'] == score['test-r2-mean'].max()][:1]


# In[ ]:





# In[68]:


xgb_model_depth = XGBRegressor(n_estimators=45, 
                               max_depth=15, 
                               eta=0.3, 
                               #subsample=0.7, 
                               #colsample_bytree=0.8,
                               n_jobs = -1,
                               random_state = XGB_RANDOMSEED,
                              )

xgb_model_depth.fit(train[train.columns[4:]], train[['depth']],
                    early_stopping_rounds=5,
                    eval_set=[(val[val.columns[4:]], val[['depth']])], 
                    verbose=False
                   )

# Get predictions and metrics
preds_train_depth = xgb_model_depth.predict(train[train.columns[4:]])
preds_val_depth   = xgb_model_depth.predict(val[train.columns[4:]])

train_score_depth = r2_score(train["depth"], preds_train_depth)
val_score_depth   = r2_score(val["depth"],   preds_val_depth)

train_score_depth, val_score_depth


# In[ ]:





# In[ ]:





# # full_reads_percent

# In[69]:


#cv_ntrees = 100

xgb_params_frp = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    #'learning_rate': 0.05,
    'eta': 0.3,
    'max_depth': 15, 
    #'num_boost_round': 10000, 
    #'early_stopping_rounds': 100,
}
dtrain = xgb.DMatrix(train[train.columns[4:]], label=train[['full_reads_percent']])


# In[70]:


get_ipython().run_cell_magic('time', '', "score = xgb.cv(xgb_params_frp, dtrain, cv_ntrees, nfold=5, #early_stopping_rounds=1000,\n        metrics={'rmse'},\n        custom_metric = r2,\n       #callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]\n      )\nscore.tail()")


# In[71]:


score[score['train-r2-mean'] == score['train-r2-mean'].max()][:1]


# In[72]:


score[score['test-r2-mean'] == score['test-r2-mean'].max()][:1]


# In[ ]:





# In[73]:


xgb_model_frp = XGBRegressor(n_estimators=45, 
                               max_depth=15, 
                               eta=0.3, 
                               #subsample=0.7, 
                               #colsample_bytree=0.8,
                               n_jobs = -1,
                               random_state = XGB_RANDOMSEED,
                              )

xgb_model_frp.fit(train[train.columns[4:]], train[['full_reads_percent']],
                    early_stopping_rounds=5,
                    eval_set=[(val[val.columns[4:]], val[['full_reads_percent']])], 
                    verbose=False
                   )

# Get predictions and metrics
preds_train_frp = xgb_model_frp.predict(train[train.columns[4:]])
preds_val_frp   = xgb_model_frp.predict(val[train.columns[4:]])

train_score_frp = r2_score(train["full_reads_percent"], preds_train_frp)
val_score_frp   = r2_score(val["full_reads_percent"],   preds_val_frp)

train_score_frp, val_score_frp


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


def plot_corrc(inp_df, inp_cols, targ_cols = ['views', 'depth', 'full_reads_percent']):
    f, ax = plt.subplots(1, 2, figsize=(24, 8))
    sns.heatmap(inp_df[inp_cols + targ_cols].corr(), 
    #sns.heatmap(inp_df.query('c2 == 0')[inp_cols + targ_cols].corr(), 
                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[0])
    sns.heatmap(inp_df[inp_cols + targ_cols].corr(method = 'spearman'), 
    #sns.heatmap(inp_df.query('c2 == 1')[inp_cols + targ_cols].corr(), 
                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[1])
#    sns.heatmap(inp_df.query('c2 == 0')[inp_cols + targ_cols].corr(method = 'spearman'), 
#                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[1, 0])
#    sns.heatmap(inp_df.query('c2 == 1')[inp_cols + targ_cols].corr(method = 'spearman'), 
#                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[1, 1])
    
    sns.pairplot(inp_df[inp_cols + targ_cols], height = 16,) #hue = 'c2')


# In[11]:


#plot_corrc(train, ['views'], ['views_pred_cb'])
#plot_corrc(train, ['views'], ['views_pred_xgb'])
#plot_corrc(train, ['views'], ['views_pred_lgb'])


# In[75]:


#plot_corrc(train, ['depth'], ['depth_pred_cb'])
plot_corrc(train, ['depth'], ['depth_pred_xgb'])
#plot_corrc(train, ['depth'], ['depth_pred_lgb'])


# In[ ]:




