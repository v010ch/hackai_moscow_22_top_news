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





# In[8]:


NTRY = 25
NAME = f'{NTRY}_xgb_pca64_sber_bord_nose_iter_2mod'


# In[9]:


VIEWS_UKR = 2554204
DEPTH_UKR = 1.799
FPR_UKR = 4.978


# In[ ]:





# ## Load data

# In[10]:


df_train  = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'))#, index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'))#, index_col= 0)
    
with open(os.path.join(DIR_DATA, 'clmns.pkl'), 'rb') as pickle_file:
    clmns = pkl.load(pickle_file)


# In[11]:


df_train.shape, df_test.shape, 


# In[12]:


cat_cols = []
num_cols = []

for el in clmns.keys():
    cat_cols.extend(clmns[el]['cat'])
    num_cols.extend(clmns[el]['num'])
    if len(clmns[el]['both']) != 0:
        print(clmns[el]['both'])


# In[13]:


num_cols.extend(['hour', 'mounth', 'dow',
                'ph_report', 'ph_gallery', 'tv_prog', 'online', 'video', 'infogr',
                 'holiday', 'day_before_holiday', 'day_after_holiday', #'distrib_brdr',
                 'two_articles',
                 #'spec_event_1',
                ])
#cat_cols.extend(['dow'])


# In[ ]:





# In[14]:


def plot_importance(inp_model, imp_number = 30, imp_type = 'weight'):
    feature_important = inp_model.get_booster().get_score(importance_type=imp_type)
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    data.nlargest(imp_number, columns="score").plot(kind='barh', figsize = (30,16)) ## plot top 40 features


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


#def r2(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
def r2(y_pred: np.ndarray, y_true: xgb.DMatrix) -> Tuple[str, float]:
    
    return 'r2', r2_score(y_true.get_label(), y_pred)


# ## views

# In[16]:


#xgb.set_config(verbosity=0)
#xgb_spec = ['day', 'mounth', 'authors_int', 'category_int']


# In[17]:


cv_ntrees = 500

xgb_params_views = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    #'learning_rate': 0.05,
    #'eta': 0.3,
    #'max_depth': 15, 
    #'num_boost_round': 10000, 
    #'early_stopping_rounds': 100,
}
#dtrain_v = xgb.DMatrix(df_train[num_cols], label=df_train[['views']])
#dtrain = xgb.DMatrix(df_train[xgb_spec], label=df_train[['views']])


# In[18]:


def get_model(inp_df, inp_params, target):
    
    dtrain = xgb.DMatrix(inp_df[num_cols], label = inp_df[[target]])
    
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

    model.fit(inp_df[num_cols], inp_df[target], 
                    verbose=False
                   )
    
    return model


# In[ ]:





# In[ ]:




%%time
scores_v = xgb.cv(xgb_params_views, dtrain_v, cv_ntrees, nfold=5, #early_stopping_rounds=1000,
        metrics={'rmse'},
        custom_metric = r2,
       #callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]
      )
#scores_v.tail(5)
# In[19]:


#if scores_v['test-rmse-mean'].argmin() != scores_v['test-r2-mean'].argmax():
#    raise ValueError('wtf?', scores_v['test-rmse-mean'].argmin(), scores_v['test-r2-mean'].argmax())

scores_v[scores_v['test-rmse-mean'] == scores_v['test-rmse-mean'].min()][:1]18	13632.72718	559.933897	43588.009905	12205.524327	0.948294	0.006664	0.450216	0.066209views_iter = scores_v['test-rmse-mean'].argmin()xgb_model_views = XGBRegressor(n_estimators=views_iter, 
                               #max_depth=7, 
                               #eta=0.1, 
                               #subsample=0.7, 
                               #colsample_bytree=0.8,
                               n_jobs = -1,
                               random_state = XGB_RANDOMSEED,
                              )

xgb_model_views.fit(df_train[num_cols], df_train['views'], 
                    verbose=False
                   )
# In[20]:


get_ipython().run_cell_magic('time', '', "model_views_start = get_model(df_train[df_train.distrib_brdr == 1], xgb_params_views, 'views')")


# In[21]:


plot_importance(model_views_start, 30, 'weight')


# In[ ]:





# In[22]:


get_ipython().run_cell_magic('time', '', "model_views_end = get_model(df_train[df_train.distrib_brdr == 0], xgb_params_views, 'views')")


# In[23]:


plot_importance(model_views_end, 30, 'weight')


# In[ ]:





# ## depth

# In[24]:


xgb_params_depth = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    #'n_estimators': 1000, 
    #'learning_rate': 0.05,
    #'eta': 0.3,
    #'max_depth': 15, 
 #   'num_boost_round': 10000, 
 #   'early_stopping_rounds': 100,
}
#dtrain_d = xgb.DMatrix(df_train[num_cols], label=df_train[['depth']])

%%time
scores_d = xgb.cv(xgb_params_depth, dtrain_d, cv_ntrees, nfold=5, #early_stopping_rounds=1000,
        metrics={'rmse'},
        custom_metric = r2,
       #callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]
      )
#scores_d.tail()
# In[25]:


#if scores_d['test-rmse-mean'].argmin() != scores_d['test-r2-mean'].argmax():
#    raise ValueError('wtf?', scores_d['test-rmse-mean'].argmin(), scores_d['test-r2-mean'].argmax())

scores_d[scores_d['test-rmse-mean'] == scores_d['test-rmse-mean'].min()][:1]36	0.000655	0.00005	    0.023424	0.000554	0.999856	0.000023	0.81588	    0.011583
23	0.001309	0.000082	0.023411	0.000574	0.999424	0.000074	0.816096	0.011584depth_iter = scores_d['test-rmse-mean'].argmin()xgb_model_depth = XGBRegressor(n_estimators=depth_iter, 
                               #max_depth=7, 
                               #eta=0.1, 
                               #subsample=0.7, 
                               #colsample_bytree=0.8,
                               n_jobs = -1,
                               random_state = XGB_RANDOMSEED,
                              )

xgb_model_depth.fit(df_train[num_cols], df_train['depth'], 
                    verbose=False
                   )
# In[26]:


get_ipython().run_cell_magic('time', '', "model_depth_start = get_model(df_train[df_train.distrib_brdr == 1], xgb_params_depth, 'depth')")


# In[27]:


plot_importance(model_depth_start, 30, 'weight')


# In[ ]:





# In[28]:


get_ipython().run_cell_magic('time', '', "model_depth_end = get_model(df_train[df_train.distrib_brdr == 0], xgb_params_depth, 'depth')")


# In[29]:


plot_importance(model_depth_end, 30, 'weight')


# In[ ]:





# ## full_reads_percent

# In[ ]:





# In[30]:


xgb_params_frp = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    #'n_estimators': 1000, 
    #'learning_rate': 0.05,
    #'eta': 0.3,
    #'max_depth': 15, 
 #   'num_boost_round': 10000, 
 #   'early_stopping_rounds': 100,
}
#dtrain_f = xgb.DMatrix(df_train[num_cols], label=df_train[['full_reads_percent']])

%%time
scores_f = xgb.cv(xgb_params_fpr, dtrain_f, cv_ntrees, nfold=5, #early_stopping_rounds=1000,
        metrics={'rmse'},
        custom_metric = r2,
       #callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]
      )
#scores_f.tail()
# In[31]:


#if scores_f['test-rmse-mean'].argmin() != scores_f['test-r2-mean'].argmax():
#    raise ValueError('wtf?', scores_f['test-rmse-mean'].argmin(), scores_f['test-r2-mean'].argmax())

scores_f[scores_f['test-rmse-mean'] == scores_f['test-rmse-mean'].min()][:1]15	4.712886	0.041195	6.837132	0.046951	0.784065	0.005105	0.544971	0.013879frp_iter = scores_f['test-rmse-mean'].argmin()xgb_model_frp = XGBRegressor(n_estimators=frp_iter,
                             #max_depth=7, 
                             #eta=0.1, 
                             #subsample=0.7, 
                             #colsample_bytree=0.8,
                             n_jobs = -1,
                             random_state = XGB_RANDOMSEED,
                             )

xgb_model_frp.fit(df_train[num_cols], df_train['full_reads_percent'],
                  verbose=False
                 )
# In[32]:


get_ipython().run_cell_magic('time', '', "model_frp_start = get_model(df_train[df_train.distrib_brdr == 1], xgb_params_frp, 'full_reads_percent')")


# In[33]:


plot_importance(model_frp_start, 30, 'weight')


# In[ ]:





# In[34]:


get_ipython().run_cell_magic('time', '', "model_frp_end = get_model(df_train[df_train.distrib_brdr == 0], xgb_params_frp, 'full_reads_percent')")


# In[35]:


plot_importance(model_frp_end, 30, 'weight')


# In[ ]:





# # Сохраняем предсказания для ансамблей / стекинга
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





# In[43]:


pred_train = pd.DataFrame()
pred_train[['document_id', 'distrib_brdr']] = df_train[['document_id', 'distrib_brdr']]
pred_train = pred_train.reindex(['document_id', 'distrib_brdr', 'views', 'depth', 'full_reads_percent'], axis = 1)


# In[44]:


pred_train.loc[pred_train.query('distrib_brdr == 1').index, 'views'] = model_views_start.predict(df_train[df_train.distrib_brdr == 1][num_cols])
pred_train.loc[pred_train.query('distrib_brdr == 0').index, 'views'] = model_views_end.predict(df_train[df_train.distrib_brdr == 0][num_cols])
print(sum(pred_train.views.isna()), ' Nan in views')


# In[45]:


pred_train.loc[pred_train.query('distrib_brdr == 1').index, 'depth'] = model_depth_start.predict(df_train[df_train.distrib_brdr == 1][num_cols])
pred_train.loc[pred_train.query('distrib_brdr == 0').index, 'depth'] = model_depth_end.predict(df_train[df_train.distrib_brdr == 0][num_cols])
print(sum(pred_train.depth.isna()), ' Nan in depth')


# In[46]:


pred_train.loc[pred_train.query('distrib_brdr == 1').index, 'full_reads_percent'] = model_frp_start.predict(df_train[df_train.distrib_brdr == 1][num_cols])
pred_train.loc[pred_train.query('distrib_brdr == 0').index, 'full_reads_percent'] = model_frp_end.predict(df_train[df_train.distrib_brdr == 0][num_cols])
print(sum(pred_train.full_reads_percent.isna()), ' Nan in full_reads_percent')


# In[47]:


pred_train.drop(['distrib_brdr'], axis =1, inplace = True)
pred_train.to_csv(os.path.join(DIR_SUBM_PART, f'{NAME}_train_part.csv'), index = False)


# In[ ]:





# In[ ]:





# ## save models

# In[41]:


model_views_start.save_model(os.path.join(DIR_MODELS, f'{NAME}_v_start.json'), 
                          )
model_views_end.save_model(os.path.join(DIR_MODELS, f'{NAME}_v_end.json'), 
                          )

model_depth_start.save_model(os.path.join(DIR_MODELS, f'{NAME}_d_start.json'), 
                          )
model_depth_end.save_model(os.path.join(DIR_MODELS, f'{NAME}_d_end.json'), 
                          )

model_frp_start.save_model(os.path.join(DIR_MODELS, f'{NAME}_f_start.json'), 
                        )
model_frp_end.save_model(os.path.join(DIR_MODELS, f'{NAME}_f_end.json'), 
                        )


# In[ ]:





# ## make predict
pred_views = xgb_model_views.predict(df_test[num_cols])
pred_depth = xgb_model_depth.predict(df_test[num_cols])
pred_frp   = xgb_model_frp.predict(  df_test[num_cols])subm = pd.DataFrame()
subm['document_id'] = df_test.document_id

subm['views'] = pred_views
subm['depth'] = pred_depth
subm['full_reads_percent'] = pred_frp
# In[48]:


subm = pd.DataFrame()
subm[['document_id', 'distrib_brdr']] = df_test[['document_id', 'distrib_brdr']]
subm = subm.reindex(['document_id', 'distrib_brdr', 'views', 'depth', 'full_reads_percent'], axis = 1)


# In[52]:


subm.loc[subm.query('distrib_brdr == 1').index, 'views'] = model_views_start.predict(df_test[df_test.distrib_brdr == 1][num_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'views'] = model_views_end.predict(df_test[df_test.distrib_brdr == 0][num_cols])
print(sum(subm.views.isna()), ' Nan in views')


# In[53]:


subm.loc[subm.query('distrib_brdr == 1').index, 'depth'] = model_depth_start.predict(df_test[df_test.distrib_brdr == 1][num_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'depth'] = model_depth_end.predict(df_test[df_test.distrib_brdr == 0][num_cols])
print(sum(subm.depth.isna()), ' Nan in depth')


# In[54]:


subm.loc[subm.query('distrib_brdr == 1').index, 'full_reads_percent'] = model_frp_start.predict(df_test[df_test.distrib_brdr == 1][num_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'full_reads_percent'] = model_frp_end.predict(df_test[df_test.distrib_brdr == 0][num_cols])
print(sum(subm.full_reads_percent.isna()), ' Nan in full_reads_percent')


# In[ ]:





# In[55]:


doc_id_ukr = df_test[df_test.spec == 1].document_id.values
subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[56]:


# присваиваем статичные данные
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'views'] = VIEWS_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'depth'] = DEPTH_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'full_reads_percent'] = FPR_UKR

subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[57]:


subm.head()


# In[58]:


subm.to_csv(os.path.join(DIR_SUBM, f'{NAME}.csv'), index = False)


# In[ ]:




