#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import time
notebookstart= time.time()


# In[2]:


import os
#import pickle as pkl

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


from xgboost import XGBRegressor
import xgboost as xgb

from sklearn import preprocessing
from sklearn.metrics import r2_score
from typing import Tuple


# In[ ]:





# In[4]:


DIR_DATA   = os.path.join(os.getcwd(), 'data')
DIR_MODELS = os.path.join(os.getcwd(), 'models')
DIR_SUBM   = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_PART = os.path.join(os.getcwd(), 'subm', 'partial')


# In[5]:


NTRY = 25
NAME_ENS = f'{NTRY}_ens_pca64_sber_nose_iter_2mod'
NAME_CB  = f'{NTRY}_cb_pca64_sber_bord_nose_iter_2mod'
NAME_XGB = f'{NTRY}_xgb_pca64_sber_bord_nose_iter_2mod'
NAME_LGB = f'{NTRY}_lgb_pca64_sber_bord_nose_iter_2mod'
#xgb_pca64_sber_lags_parse_bord_nose_val_part


# In[6]:


VIEWS_UKR = 2554204
DEPTH_UKR = 1.799
FPR_UKR = 4.978


# In[7]:


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
#subm_views = pd.read_csv(os.path.join(DIR_SUBM, f'1_xgb_baseline_test.csv'),  usecols=['document_id', 'views'])
subm_views = pd.read_csv(os.path.join(DIR_SUBM, f'{NTRY}_lgb_pca64_sber_bord_nose_iter_poly.csv'),  usecols=['document_id', 'views'])
subm_depth = pd.read_csv(os.path.join(DIR_SUBM, f'{NTRY}_lgb_pca64_sber_bord_nose_iter_poly.csv'), usecols=['document_id', 'depth'])
subm_frp   = pd.read_csv(os.path.join(DIR_SUBM, f'{NTRY}_lgb_pca64_sber_bord_nose_iter_poly.csv'), usecols=['document_id', 'full_reads_percent'])

#ubm_depth = pd.read_csv(os.path.join(DIR_SUBM, f'{NTRY}_lgb_pca64_sber_lags_parse_bord_nose_irq.csv'), usecols=['document_id', 'depth'])
#ubm_frp   = pd.read_csv(os.path.join(DIR_SUBM, f'14_lgb_pca64_sber_lags_parse_bord_nose.csv'),   usecols=['document_id', 'full_reads_percent'])

print(subm_views.shape, subm_depth.shape, subm_frp.shape)subm = subm_views.merge(subm_depth, on='document_id', validate='one_to_one')
#subm = subm.merge(subm_depth, on='document_id', validate='one_to_one')
subm = subm.merge(subm_frp, on='document_id', validate='one_to_one')subm.to_csv(os.path.join(DIR_SUBM, f'{NTRY}_concatenate.csv'), index = False)
# In[ ]:





# In[ ]:





# # Ансамбли

# In[8]:


def plot_importance(inp_model, imp_number = 30, imp_type = 'weight'):
    feature_important = inp_model.get_booster().get_score(importance_type=imp_type)
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    data.nlargest(imp_number, columns="score").plot(kind='barh', figsize = (30,16)) ## plot top 40 features


# In[9]:


train = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'), usecols = ['document_id', 'views', 'depth', 'full_reads_percent', 'distrib_brdr', ])
test  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'),  usecols = ['document_id', 'distrib_brdr', 'spec',])

cb_train = pd.read_csv(os.path.join(DIR_SUBM_PART, f'{NAME_CB}_train_part.csv'))
cb_test  = pd.read_csv(os.path.join(DIR_SUBM, f'{NAME_CB}.csv'))

xgb_train = pd.read_csv(os.path.join(DIR_SUBM_PART, f'{NAME_XGB}_train_part.csv'))
xgb_test  = pd.read_csv(os.path.join(DIR_SUBM, f'{NAME_XGB}.csv'))

lgb_train = pd.read_csv(os.path.join(DIR_SUBM_PART, f'{NAME_LGB}_train_part.csv'))
lgb_test  = pd.read_csv(os.path.join(DIR_SUBM, f'{NAME_LGB}.csv'))


# In[10]:


test.head()


# In[11]:


#train.head(3)
#test.head(3)


# In[12]:


print('before ', train.shape, cb_train.shape, lgb_train.shape, xgb_train.shape)
print('before ', test.shape, cb_test.shape, lgb_test.shape, xgb_test.shape)
train = pd.concat([train.reset_index(drop = True),
                   cb_train.reset_index(drop = True),
                   xgb_train.reset_index(drop = True),
                   lgb_train.reset_index(drop = True),
                  ], ignore_index = True, axis = 1)

test = pd.concat([test.reset_index(drop = True),
                   cb_test.reset_index(drop = True),
                   xgb_test.reset_index(drop = True),
                   lgb_test.reset_index(drop = True),
                  ], ignore_index = True, axis = 1)

train_cols = ['document_id', 'views', 'depth', 'full_reads_percent', 'distrib_brdr',
         'document_id_cb', 'views_pred_cb', 'depth_pred_cb', 'frp_pred_cb',
         'document_id_xgb', 'views_pred_xgb', 'depth_pred_xgb', 'frp_pred_xgb',
         'document_id_lgb', 'views_pred_lgb', 'depth_pred_lgb', 'frp_pred_lgb',
        ]
test_cols = ['document_id', 'distrib_brdr', 'spec',
         'document_id_cb', 'views_pred_cb', 'depth_pred_cb', 'frp_pred_cb',
         'document_id_xgb', 'views_pred_xgb', 'depth_pred_xgb', 'frp_pred_xgb',
         'document_id_lgb', 'views_pred_lgb', 'depth_pred_lgb', 'frp_pred_lgb',
        ]
train.columns = train_cols
test.columns  = test_cols


if sum(train.document_id == train.document_id_xgb) != train.shape[0] or    sum(train.document_id == train.document_id_lgb) != train.shape[0]:
    print('wtf train')
    
if sum(test.document_id == test.document_id_xgb) != test.shape[0] or    sum(test.document_id == test.document_id_lgb) != test.shape[0]:
    print('wtf test')    
    
train.drop(['document_id_cb', 'document_id_xgb', 'document_id_lgb'], axis = 1, inplace = True)
test.drop(['document_id_cb', 'document_id_xgb', 'document_id_lgb'], axis = 1, inplace = True)
   
print('after ', train.shape)
print('after ', test.shape)


# In[13]:


#train.columns[5:]
test.columns


# In[14]:


test.head(5)


# # нормализуем

# In[15]:


train_cols = train.columns[5:]
print(train_cols)


# In[16]:


scaler_start = preprocessing.StandardScaler()  #Standardize features by removing the mean and scaling to unit variance.
scaler_end = preprocessing.StandardScaler()  #Standardize features by removing the mean and scaling to unit variance.

scaler_start.fit(train[train.distrib_brdr == 1][train_cols])
scaler_end.fit(train[train.distrib_brdr == 0][train_cols])

train.loc[train.query('distrib_brdr == 1').index, train_cols] = scaler_start.transform(train[train.distrib_brdr == 1][train_cols])
test.loc[test.query('distrib_brdr == 1').index, train_cols]  = scaler_start.transform(test[test.distrib_brdr == 1][train_cols])

train.loc[train.query('distrib_brdr == 0').index, train_cols] = scaler_end.transform(train[train.distrib_brdr == 0][train_cols])
test.loc[test.query('distrib_brdr == 0').index, train_cols]  = scaler_end.transform(test[test.distrib_brdr == 0][train_cols])


# In[17]:


#!!!!!!!!! нормализовать предикты
#!!!!!!!!! сделать предикты


# In[ ]:





# In[ ]:





# In[18]:


#def r2(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
def r2(y_pred: np.ndarray, y_true: xgb.DMatrix) -> Tuple[str, float]:
    
    return 'r2', r2_score(y_true.get_label(), y_pred)


# # views

# In[19]:


cv_ntrees = 100

xgb_params_views = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    #'learning_rate': 0.05,
    'eta': 0.3,
    'max_depth': 4, 
    #'lambda': 10,
    #'num_boost_round': 10000, 
    #'early_stopping_rounds': 100,
}
#dtrain = xgb.DMatrix(train[train.columns[4:]], label=train[['views']])
#dval   = xgb.DMatrix(val[train.columns[4:]], label=val[['views']])


# In[20]:


def get_model(inp_df, inp_params, target):
    
    dtrain = xgb.DMatrix(inp_df[train_cols], label = inp_df[[target]])
    
    scores = xgb.cv(inp_params, dtrain, cv_ntrees, nfold=5, #early_stopping_rounds=1000,
        metrics={'rmse'},
        custom_metric = r2,
       #callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]
      )
    
    print(scores[scores['test-rmse-mean'] == scores['test-rmse-mean'].min()][:1].to_string())
    niters = scores['test-rmse-mean'].argmin()
    
    model = XGBRegressor(n_estimators=niters, 
                               #max_depth=7, 
                               #eta=0.1, 
                               #subsample=0.7, 
                               #colsample_bytree=0.8,
                               
                               n_jobs = -1,
                               random_state = XGB_RANDOMSEED,
                              )

    model.fit(inp_df[train_cols], inp_df[target], 
                    verbose=False
                   )
    
    return model

%%time
score = xgb.cv(xgb_params_views, dtrain, cv_ntrees, nfold=5, #early_stopping_rounds=1000,
        metrics={'rmse'},
        custom_metric = r2,
       #callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]
      )
score.tail()score[score['train-r2-mean'] == score['train-r2-mean'].max()][:1]score[score['test-r2-mean'] == score['test-r2-mean'].max()][:1]
# In[ ]:





# In[21]:


get_ipython().run_cell_magic('time', '', "model_views_start = get_model(train[train.distrib_brdr == 1], xgb_params_views, 'views')")


# In[22]:


plot_importance(model_views_start, 8, 'weight')


# In[ ]:





# In[23]:


get_ipython().run_cell_magic('time', '', "model_views_end = get_model(train[train.distrib_brdr == 0], xgb_params_views, 'views')")


# In[24]:


plot_importance(model_views_end, 8, 'weight')


# In[ ]:





# In[ ]:





# # depth

# In[25]:


#cv_ntrees = 100

xgb_params_depth = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    #'learning_rate': 0.05,
    'eta': 0.3,
    'max_depth': 4, 
    #'num_boost_round': 10000, 
    #'early_stopping_rounds': 100,
}
#dtrain = xgb.DMatrix(train[train.columns[4:]], label=train[['depth']])

%%time
score = xgb.cv(xgb_params_depth, dtrain, cv_ntrees, nfold=5, #early_stopping_rounds=1000,
        metrics={'rmse'},
        custom_metric = r2,
       #callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]
      )
score.tail()score[score['train-r2-mean'] == score['train-r2-mean'].max()][:1]score[score['test-r2-mean'] == score['test-r2-mean'].max()][:1]xgb_model_depth = XGBRegressor(n_estimators=45, 
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





# In[26]:


get_ipython().run_cell_magic('time', '', "model_depth_start = get_model(train[train.distrib_brdr == 1], xgb_params_depth, 'depth')")


# In[27]:


plot_importance(model_depth_start, 8, 'weight')


# In[ ]:





# In[28]:


get_ipython().run_cell_magic('time', '', "model_depth_end = get_model(train[train.distrib_brdr == 0], xgb_params_depth, 'depth')")


# In[29]:


plot_importance(model_depth_end, 8, 'weight')


# In[ ]:





# # full_reads_percent

# In[30]:


#cv_ntrees = 100

xgb_params_frp = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    #'learning_rate': 0.05,
    'eta': 0.3,
    'max_depth': 4, 
    #'num_boost_round': 10000, 
    #'early_stopping_rounds': 100,
}
#dtrain = xgb.DMatrix(train[train.columns[4:]], label=train[['full_reads_percent']])

%%time
score = xgb.cv(xgb_params_frp, dtrain, cv_ntrees, nfold=5, #early_stopping_rounds=1000,
        metrics={'rmse'},
        custom_metric = r2,
       #callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]
      )
score.tail()score[score['train-r2-mean'] == score['train-r2-mean'].max()][:1]score[score['test-r2-mean'] == score['test-r2-mean'].max()][:1]
# In[ ]:




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





# In[31]:


get_ipython().run_cell_magic('time', '', "model_frp_start = get_model(train[train.distrib_brdr == 1], xgb_params_frp, 'full_reads_percent')")


# In[32]:


plot_importance(model_frp_start, 8, 'weight')


# In[ ]:





# In[33]:


get_ipython().run_cell_magic('time', '', "model_frp_end = get_model(train[train.distrib_brdr == 0], xgb_params_frp, 'full_reads_percent')")


# In[34]:


plot_importance(model_frp_end, 8, 'weight')


# In[ ]:





# In[ ]:





# In[ ]:





# # Сохраняем финальные модели

# In[35]:


model_views_start.save_model(os.path.join(DIR_MODELS, f'{NAME_ENS}_v_start.json'), 
                          )
model_views_end.save_model(os.path.join(DIR_MODELS, f'{NAME_ENS}_v_end.json'), 
                          )

model_depth_start.save_model(os.path.join(DIR_MODELS, f'{NAME_ENS}_d_start.json'), 
                          )
model_depth_end.save_model(os.path.join(DIR_MODELS, f'{NAME_ENS}_d_end.json'), 
                          )

model_frp_start.save_model(os.path.join(DIR_MODELS, f'{NAME_ENS}_f_start.json'), 
                        )
model_frp_end.save_model(os.path.join(DIR_MODELS, f'{NAME_ENS}_f_end.json'), 
                        )


# In[ ]:





# In[ ]:





# # Делаем предсказание

# In[ ]:





# In[36]:


subm = pd.DataFrame()
subm[['document_id', 'distrib_brdr']] = test[['document_id', 'distrib_brdr']]
subm = subm.reindex(['document_id', 'distrib_brdr', 'views', 'depth', 'full_reads_percent'], axis = 1)


# In[37]:


subm.loc[subm.query('distrib_brdr == 1').index, 'views'] = model_views_start.predict(test[test.distrib_brdr == 1][train_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'views'] = model_views_end.predict(test[test.distrib_brdr == 0][train_cols])
print(sum(subm.views.isna()), ' Nan in views')


# In[38]:


subm.loc[subm.query('distrib_brdr == 1').index, 'depth'] = model_depth_start.predict(test[test.distrib_brdr == 1][train_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'depth'] = model_depth_end.predict(test[test.distrib_brdr == 0][train_cols])
print(sum(subm.depth.isna()), ' Nan in depth')


# In[39]:


subm.loc[subm.query('distrib_brdr == 1').index, 'full_reads_percent'] = model_frp_start.predict(test[test.distrib_brdr == 1][train_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'full_reads_percent'] = model_frp_end.predict(test[test.distrib_brdr == 0][train_cols])
print(sum(subm.full_reads_percent.isna()), ' Nan in full_reads_percent')


# In[ ]:





# In[40]:


doc_id_ukr = test[test.spec == 1].document_id.values
subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[41]:


# присваиваем статичные данные
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'views'] = VIEWS_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'depth'] = DEPTH_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'full_reads_percent'] = FPR_UKR

subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[42]:


subm.head()


# In[43]:


subm.drop(['distrib_brdr'], inplace = True, axis = 1)
subm.to_csv(os.path.join(DIR_SUBM, f'{NAME_ENS}.csv'), index = False)


# In[45]:


subm.head()


# In[44]:


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




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


# In[ ]:


#plot_corrc(train, ['depth'], ['depth_pred_cb'])
#plot_corrc(train, ['depth'], ['depth_pred_xgb'])
#plot_corrc(train, ['depth'], ['depth_pred_lgb'])


# In[ ]:




