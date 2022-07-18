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

from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn import preprocessing

from typing import Tuple


# In[4]:


from xgboost import __version__ as xgb_version
from sklearn import __version__ as sklearn_version

print(f'xgb_version: {xgb_version}')
print(f'sklearn_version: {sklearn_version}')


# In[5]:


get_ipython().run_line_magic('watermark', '--iversions')


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

CB_RANDOMSEED = 309487
XGB_RANDOMSEED = 56


# In[ ]:





# In[7]:


DIR_DATA   = os.path.join(os.getcwd(), 'data')
DIR_MODELS = os.path.join(os.getcwd(), 'models')
DIR_SUBM   = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_PART = os.path.join(os.getcwd(), 'subm', 'partial')


# In[ ]:





# In[8]:


NTRY = 15
NAME = f'{NTRY}_xgb_pca64_sber_lags_parse_bord_nose_irq'


# In[9]:


CTR_UKR = 6.096

VIEWS_UKR = 2554204
DEPTH_UKR = 1.799
FPR_UKR = 4.978


# In[ ]:





# ## Load data

# In[10]:


df_train  = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'), index_col= 0)
x_train  = pd.read_csv(os.path.join(DIR_DATA, 'x_train.csv'), index_col= 0)
x_val    = pd.read_csv(os.path.join(DIR_DATA, 'x_val.csv'), index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'), index_col= 0)

with open(os.path.join(DIR_DATA, 'cat_columns.pkl'), 'rb') as pickle_file:
    cat_cols = pkl.load(pickle_file)
    
with open(os.path.join(DIR_DATA, 'num_columns.pkl'), 'rb') as pickle_file:
    num_cols = pkl.load(pickle_file)
    
with open(os.path.join(DIR_DATA, 'clmns.pkl'), 'rb') as pickle_file:
    clmns = pkl.load(pickle_file)


# In[11]:


x_train.shape, x_val.shape, df_test.shape, len(cat_cols), len(num_cols)


# отделяем метки от данных

# In[12]:


y_train = x_train[['views', 'depth', 'full_reads_percent']]
y_val   = x_val[['views', 'depth', 'full_reads_percent']]

x_train.drop(['views', 'depth', 'full_reads_percent'], axis = 1, inplace = True)
x_val.drop(  ['views', 'depth', 'full_reads_percent'], axis = 1, inplace = True)

x_train.shape, x_val.shape, y_train.shape, y_val.shape


# In[13]:


#cat_cols + num_cols


# In[14]:


cat_cols = []
num_cols = []

for el in clmns.keys():
    cat_cols.extend(clmns[el]['cat'])
    num_cols.extend(clmns[el]['num'])
    if len(clmns[el]['both']) != 0:
        print(clmns[el]['both'])


# In[15]:


num_cols.extend(['hour', 'mounth', 'dow',
                'ph_report', 'ph_gallery', 'tv_prog', 'online', 'video', 'infogr',
                 'holiday', 'day_before_holiday', 'day_after_holiday', 'distrib_brdr',
                 'two_articles',
                 #'spec_event_1',
                ])
#cat_cols.extend(['dow'])


# In[ ]:





# In[16]:


def plot_importance(inp_model, imp_number = 30, imp_type = 'weight'):
    feature_important = inp_model.get_booster().get_score(importance_type=imp_type)
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    data.nlargest(imp_number, columns="score").plot(kind='barh', figsize = (30,16)) ## plot top 40 features


# In[ ]:





# In[ ]:





# In[17]:


cv_ntrees = 100


# In[18]:


#def r2(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
def r2(y_pred: np.ndarray, y_true: xgb.DMatrix) -> Tuple[str, float]:
    
    #preds = np.array(y_pred[0])
    #print(type(y_true))
    #print(type(y_pred)) # np.array
    
    return 'r2', r2_score(y_true.get_label(), y_pred)
    


# ## views

# In[19]:


#xgb.set_config(verbosity=0)


# In[20]:


cb_params_views = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    #'learning_rate': 0.05,
    'eta': 0.3,
    'max_depth': 15, 
    #'num_boost_round': 10000, 
    #'early_stopping_rounds': 100,
}
dtrain = xgb.DMatrix(df_train[num_cols], label=df_train[['views']])


# In[21]:


get_ipython().run_cell_magic('time', '', "score = xgb.cv(cb_params_views, dtrain, cv_ntrees, nfold=5, #early_stopping_rounds=1000,\n        metrics={'rmse'},\n        custom_metric = r2,\n       #callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]\n      )")


# In[22]:


score.tail(5)


# In[23]:


score[score['train-r2-mean'] == score['train-r2-mean'].max()][:1]


# In[24]:


score[score['test-r2-mean'] == score['test-r2-mean'].max()][:1]

55	3541.342715	    641.230136	49545.854791	12187.893784	0.996433	0.001161	0.277382	0.11142
8	10656.078996	1508.117669	49275.311294	12570.20743	    0.968962	0.004469	0.287494	0.112323
# In[25]:


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





# In[26]:


# Get predictions and metrics
preds_train_views = xgb_model_views.predict(x_train[num_cols])
preds_val_views   = xgb_model_views.predict(x_val[num_cols])

train_score_views = r2_score(y_train["views"], preds_train_views)
val_score_views   = r2_score(y_val["views"],   preds_val_views)

train_score_views, val_score_views

(0.7755792099701974, 0.8464698702781239) baseline
# In[ ]:





# In[27]:


plot_importance(xgb_model_views, 30, 'weight')


# In[ ]:





# ## depth

# In[ ]:





# In[28]:


cb_params_depth = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    #'n_estimators': 1000, 
    #'learning_rate': 0.05,
    'eta': 0.3,
    'max_depth': 15, 
 #   'num_boost_round': 10000, 
 #   'early_stopping_rounds': 100,
}
dtrain = xgb.DMatrix(df_train[num_cols], label=df_train[['depth']])


# In[29]:


get_ipython().run_cell_magic('time', '', "score = xgb.cv(cb_params_depth, dtrain, cv_ntrees, nfold=5, #early_stopping_rounds=1000,\n        metrics={'rmse'},\n        custom_metric = r2,\n       #callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]\n      )\nscore.tail()")


# In[30]:


score[score['train-r2-mean'] == score['train-r2-mean'].max()][:1]


# In[31]:


score[score['test-r2-mean'] == score['test-r2-mean'].max()][:1]

41	0.000904	0.000214	0.028241	0.001703	0.999773	0.000093	0.790149	0.021323
35	0.001028	0.000131	0.02824	    0.001704	0.999718	0.000071	0.790173	0.021314
# In[32]:


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


# In[52]:


# Get predictions and metrics
preds_train_depth = xgb_model_depth.predict(x_train[num_cols])
preds_val_depth   = xgb_model_depth.predict(x_val[num_cols])

train_score_depth = r2_score(y_train["depth"], preds_train_depth)
val_score_depth   = r2_score(y_val["depth"],   preds_val_depth)

train_score_depth, val_score_depth

(0.9400349544217423, 0.804401115229307)  emb + pca 64 + lags + nauth + all_norm + parse
# In[ ]:





# In[34]:


plot_importance(xgb_model_depth, 30, 'weight')


# In[ ]:





# ## full_reads_percent

# In[ ]:





# In[35]:


cb_params_fpr = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    #'n_estimators': 1000, 
    #'learning_rate': 0.05,
    'eta': 0.3,
    'max_depth': 15, 
 #   'num_boost_round': 10000, 
 #   'early_stopping_rounds': 100,
}
dtrain = xgb.DMatrix(df_train[num_cols], label=df_train[['full_reads_percent']])


# In[36]:


get_ipython().run_cell_magic('time', '', "score = xgb.cv(cb_params_fpr, dtrain, cv_ntrees, nfold=5, #early_stopping_rounds=1000,\n        metrics={'rmse'},\n        custom_metric = r2,\n       #callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]\n      )\nscore.tail()")


# In[37]:


score[score['train-r2-mean'] == score['train-r2-mean'].max()][:1]


# In[38]:


score[score['test-r2-mean'] == score['test-r2-mean'].max()][:1]

52	0.096801	0.029933	7.717025	0.133041	0.999901	0.000048	0.420521	0.015615
47	0.097153	0.029723	7.716983	0.133039	0.9999	    0.000047	0.420528	0.015609
# In[39]:


#pd.DataFrame(preds_train_depth, columns = ['depth_pred'])
pred_scaler = preprocessing.StandardScaler()
tmp = pred_scaler.fit_transform(preds_train_depth.reshape(-1, 1))
pred_depth_train = pd.DataFrame(tmp, columns = ['depth_pred'])

pred_depth_val   = pd.DataFrame(pred_scaler.transform(preds_val_depth.reshape(-1, 1)), columns = ['depth_pred'])

print('before ', x_train.shape, x_val.shape, preds_train_depth.shape, preds_val_depth.shape)
x_train = pd.concat([x_train, pred_depth_train], axis = 1)
x_val   = pd.concat([x_val,   pred_depth_val],   axis = 1)
print('after  ', x_train.shape, x_val.shape)
# In[40]:


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


# In[41]:


# Get predictions and metrics
preds_train_frp = xgb_model_frp.predict(x_train[num_cols])
preds_val_frp   = xgb_model_frp.predict(x_val[num_cols])

train_score_frp = r2_score(y_train["full_reads_percent"], preds_train_frp)
val_score_frp   = r2_score(y_val["full_reads_percent"],   preds_val_frp)

train_score_frp, val_score_frp

(0.8954773259992533, 0.545897622350289) emb + pca 64 + lags + nauth + all_norm + parse
# In[ ]:





# In[42]:


plot_importance(xgb_model_frp, 30, 'weight')


# In[ ]:





# In[43]:


score_train = 0.4 * train_score_views + 0.3 * train_score_depth + 0.3 * train_score_frp
score_val   = 0.4 * val_score_views   + 0.3 * val_score_depth   + 0.3 * val_score_frp

score_train, score_val

(0.9298327062037938, 0.5924959277458732) emb + pca 64 + lags + nauth + all_norm + parse
# In[ ]:





# # Сохраняем предсказания для ансамблей / стекинга

# In[44]:


x_train_pred = x_train[['document_id']]
x_val_pred   = x_val[['document_id']]

print('before ', x_train_pred.shape, preds_train_views.shape, preds_train_depth.shape, preds_train_frp.shape)
print('before ', x_val_pred.shape,   preds_val_views.shape,   preds_val_depth.shape,   preds_val_frp.shape)

# https://github.com/pandas-dev/pandas/issues/25349
x_train_pred = pd.concat([x_train_pred.reset_index(drop=True), 
                          pd.DataFrame(preds_train_views).reset_index(drop = True), 
                          pd.DataFrame(preds_train_depth).reset_index(drop = True), 
                          pd.DataFrame(preds_train_frp).reset_index(drop = True)
                         ], ignore_index = True, axis = 1)
x_val_pred   = pd.concat([x_val_pred.reset_index(drop=True),   
                          pd.DataFrame(preds_val_views).reset_index(drop = True), 
                          pd.DataFrame(preds_val_depth).reset_index(drop = True), 
                          pd.DataFrame(preds_val_frp).reset_index(drop = True)
                         ], ignore_index = True, axis = 1)

x_train_pred.columns = ['document_id', 'views_pred_xgb', 'depth_pred_xgb', 'frp_pred_xgb']
x_val_pred.columns   = ['document_id', 'views_pred_xgb', 'depth_pred_xgb', 'frp_pred_xgb']

print('after ', x_train_pred.shape)
print('after ', x_val_pred.shape)

x_train_pred.to_csv(os.path.join(DIR_SUBM_PART, f'{NAME}_train_part.csv'), index = False)
x_val_pred.to_csv(os.path.join(DIR_SUBM_PART, f'{NAME}_val_part.csv'), index = False)


# In[ ]:





# In[ ]:





# ## save models

# In[45]:


xgb_model_views.save_model(os.path.join(DIR_MODELS, f'{NAME}_v.json'), 
                          )

xgb_model_depth.save_model(os.path.join(DIR_MODELS, f'{NAME}_d.json'), 
                          )

xgb_model_frp.save_model(os.path.join(DIR_MODELS, f'{NAME}_f.json'), 
                        )


# In[ ]:





# ## make predict

# In[46]:


pred_views = xgb_model_views.predict(df_test[num_cols])
pred_depth = xgb_model_depth.predict(df_test[num_cols])
pred_frp   = xgb_model_frp.predict(  df_test[num_cols])


# In[47]:


subm = pd.DataFrame()
subm['document_id'] = df_test.document_id

subm['views'] = pred_views
subm['depth'] = pred_depth
subm['full_reads_percent'] = pred_frp


# In[48]:


doc_id_ukr = df_test[df_test.ctr == CTR_UKR].document_id.values
subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[49]:


# присваиваем статичные данные
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'views'] = VIEWS_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'depth'] = DEPTH_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'full_reads_percent'] = FPR_UKR

subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[50]:


subm.head()


# In[51]:


subm.to_csv(os.path.join(DIR_SUBM, f'{NAME}.csv'), index = False)


# In[ ]:




