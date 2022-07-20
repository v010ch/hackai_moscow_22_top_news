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
from sklearn import preprocessing
from catboost import CatBoostRegressor, Pool, cv

import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


from catboost import __version__ as cb_version
from sklearn import __version__ as sklearn_version

print(f'cb_version: {cb_version}')
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


NTRY = 24
NAME = f'{NTRY}_cb_pca64_sber_bord_nose_iter_poly'


# In[9]:


VIEWS_UKR = 2554204
DEPTH_UKR = 1.799
FPR_UKR = 4.978


# In[ ]:





# ## Load data

# In[10]:


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'), index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'), index_col= 0)

with open(os.path.join(DIR_DATA, 'clmns.pkl'), 'rb') as pickle_file:
    clmns = pkl.load(pickle_file)


# In[11]:


df_train.shape, df_test.shape, 


# In[ ]:





# In[ ]:





# In[12]:


cat_cols = []
num_cols = []

for el in clmns.keys():
    cat_cols.extend(clmns[el]['cat'])
    num_cols.extend(clmns[el]['num'])
    if len(clmns[el]['both']) != 0:
        print(clmns[el]['both'])


# In[13]:


num_cols.extend(['hour', 'mounth'])
cat_cols.extend(['dow', 
                 'ph_report', 'ph_gallery', 'tv_prog', 'online', 'video', 'infogr',
                 'holiday', 'day_before_holiday', 'day_after_holiday', 'distrib_brdr',
                 #'spec_event_1',
                ])


# In[14]:


for el in cat_cols:
    df_train[el] = df_train[el].astype(str)
    df_test[el] = df_test[el].astype(str)


# In[ ]:





# In[15]:


#views
train_views_full = Pool(df_train[cat_cols + num_cols],
                      df_train[['views']],
                      cat_features = cat_cols,
                      #feature_names = cat_cols + num_cols
                     )

#depth
train_depth_full = Pool(df_train[cat_cols + num_cols],
                      df_train[['depth']],
                      cat_features = cat_cols,
                      #feature_names = cat_cols + num_cols
                     )

#frp
train_frp_full = Pool(df_train[cat_cols + num_cols],
                      df_train[['full_reads_percent']],
                      cat_features = cat_cols,
                      #feature_names = cat_cols + num_cols
                     )

#full_reads_percent
#у frp корреляция с depth. так что добавим признак deprh_pred и соберем датасет уже после предсказания depth


# In[ ]:





# In[16]:


def plot_feature_importance2(inp_model, inp_pool, imp_number = 30):
    
    data = pd.DataFrame({'feature_importance': inp_model.get_feature_importance(inp_pool), 
              'feature_names': inp_pool.get_feature_names()}).sort_values(by=['feature_importance'], 
                                                       ascending=True)
    
    data.nlargest(imp_number, columns="feature_importance").plot(kind='barh', figsize = (30,16)) ## plot top 40 features


# In[17]:


def plot_feature_importance(importance,names,model_type, imp_number = 30):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'][:imp_number], y=fi_df['feature_names'][:imp_number])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


# In[ ]:





# ## views

# In[18]:


cb_params_views = {"iterations": 100,
                  #"depth": 2,
                  "loss_function": "RMSE",
                  'eval_metric': 'R2',
                  "verbose": False
                  }


# In[19]:


get_ipython().run_cell_magic('time', '', 'scores_views = cv(train_views_full,\n                  cb_params_views,\n                  fold_count=5,\n                  seed = CB_RANDOMSEED, \n                  #plot="True"\n                 )')


# In[20]:


#scores_views


# In[52]:


if scores_views['test-RMSE-mean'].argmin() != scores_views['test-R2-mean'].argmax():
    raise ValueError('wtf?', scores_views['test-RMSE-mean'].argmin(), scores_views['test-R2-mean'].argmax())


# In[22]:


scores_views[scores_views['test-R2-mean'] == scores_views['test-R2-mean'].max()]
#@ores_views.iloc[scores_views['test-R2-mean'].argmax()]

996	996	0.592591	0.095643	0.952982	0.004997	38079.381052	13904.016339	13045.641422	600.200488
# In[ ]:





# In[23]:


views_iter = scores_views['test-R2-mean'].argmax()
print(views_iter)


# In[24]:


cb_model_views = CatBoostRegressor(iterations=views_iter,
                                 #learning_rate=0.05,
                                 #depth=10,
                                 random_seed = CB_RANDOMSEED,
                                 #n_estimators=100,
                                  )
# Fit model
cb_model_views.fit(train_views_full,
                  #plot = True,
                   verbose = 100,
                  )


# In[25]:


#plot_feature_importance(cb_model_views, train_ds_views, 30)
plot_feature_importance(cb_model_views.get_feature_importance(), train_views_full.get_feature_names(), 'CATBOOST')


# In[ ]:





# ## depth

# In[26]:


cb_params_depth = cb_params_views


# In[27]:


get_ipython().run_cell_magic('time', '', 'scores_depth = cv(train_depth_full,\n                  cb_params_depth,\n                  fold_count=5,\n                  seed = CB_RANDOMSEED, \n                  #plot="True"\n                 )')


# In[49]:


if scores_depth['test-RMSE-mean'].argmin() != scores_depth['test-R2-mean'].argmax():
    raise ValueError('wtf?', scores_depth['test-RMSE-mean'].argmin(), scores_depth['test-R2-mean'].argmax())


# In[29]:


scores_depth[scores_depth['test-R2-mean'] == scores_depth['test-R2-mean'].max()]
#scores_depth.iloc[scores_depth['test-R2-mean'].argmax()]

998	998	0.802018	0.017199	0.938263	0.00201	0.027449	0.001461	0.015352	0.000314
# In[30]:


depth_iter =  scores_depth['test-R2-mean'].argmax()
print(depth_iter)


# In[31]:


cb_model_depth = CatBoostRegressor(iterations=depth_iter,
                                 #learning_rate=0.05,
                                 #depth=10,
                                 random_seed = CB_RANDOMSEED,
                                 #n_estimators=100,
                                  )
# Fit model
cb_model_depth.fit(train_depth_full,
                   #plot = True,
                   verbose = 100,
                  )


# In[32]:


#plot_feature_importance(cb_model_views, train_ds_views, 30)
plot_feature_importance(cb_model_depth.get_feature_importance(), train_depth_full.get_feature_names(), 'CATBOOST')


# In[ ]:





# ## full_reads_percent

# In[33]:


cb_params_frp = cb_params_views


# In[34]:


get_ipython().run_cell_magic('time', '', 'scores_frp = cv(train_frp_full,\n                  cb_params_frp,\n                  fold_count=5,\n                  seed = CB_RANDOMSEED, \n                  #plot="True"\n                 )')


# In[36]:


if scores_frp['test-RMSE-mean'].argmin() != scores_frp['test-R2-mean'].argmax():
    raise ValueError('wtf?', scores_frp['test-RMSE-mean'].argmin(), scores_frp['test-R2-mean'].argmax())


# In[37]:


scores_frp[scores_frp['test-R2-mean'] == scores_frp['test-R2-mean'].max()]
#scores_frp.iloc[scores_frp['test-R2-mean'].argmax()]

999	999	0.579495	0.008608	0.820714	0.003633	6.572665	0.196964	4.294636	0.058588
# In[38]:


frp_iter = scores_frp['test-R2-mean'].argmax()
print(frp_iter)


# In[39]:


cb_model_frp = CatBoostRegressor(iterations=frp_iter,
                                 #learning_rate=0.05,
                                 #depth=10,
                                 random_seed = CB_RANDOMSEED,
                                 #n_estimators=100,
    #num_trees=None,
                                )
# Fit model
cb_model_frp.fit(train_frp_full,#train_ds_frp,
                   #eval_set=val_ds_frp, 
                   #plot = True,
                 verbose = 100,
                  )


# In[40]:


#plot_feature_importance(cb_model_views, train_ds_views, 30)
plot_feature_importance(cb_model_frp.get_feature_importance(), train_frp_full.get_feature_names(), 'CATBOOST')


# In[ ]:





# In[ ]:





# ## Сохраняем предсказания для ансамблей / стекинга
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

x_train_pred.columns = ['document_id', 'views_pred_cb', 'depth_pred_cb', 'frp_pred_cb']
x_val_pred.columns   = ['document_id', 'views_pred_cb', 'depth_pred_cb', 'frp_pred_cb']

print('after ', x_train_pred.shape)
print('after ', x_val_pred.shape)

x_train_pred.to_csv(os.path.join(DIR_SUBM_PART, f'{NAME}_train_part.csv'), index = False)
x_val_pred.to_csv(os.path.join(DIR_SUBM_PART, f'{NAME}_val_part.csv'), index = False)
# In[ ]:





# In[ ]:





# In[ ]:





# ## save models

# In[ ]:





# In[41]:


#cb_model_views.save_model(os.path.join(DIR_MODELS, f'{NTRY}_pca64_cb_views.cbm'), 
cb_model_views.save_model(os.path.join(DIR_MODELS, f'{NAME}_v.cbm'), 
                           format="cbm",
                           export_parameters=None,
                           pool=None
                         )

cb_model_depth.save_model(os.path.join(DIR_MODELS, f'{NAME}_d.cbm'), 
                           format="cbm",
                           export_parameters=None,
                           pool=None
                         )

cb_model_frp.save_model(os.path.join(DIR_MODELS, f'{NAME}_f.cbm'), 
                           format="cbm",
                           export_parameters=None,
                           pool=None
                         )


# In[ ]:





# In[ ]:





# In[ ]:





# ## make predict

# In[42]:


pred_views = cb_model_views.predict(df_test[cat_cols + num_cols])
pred_depth = cb_model_depth.predict(df_test[cat_cols + num_cols])
pred_frp   = cb_model_frp.predict(  df_test[cat_cols + num_cols])


# In[43]:


subm = pd.DataFrame()
subm['document_id'] = df_test.document_id

subm['views'] = pred_views
subm['depth'] = pred_depth
subm['full_reads_percent'] = pred_frp


# In[44]:


doc_id_ukr = df_test[df_test.spec == 1].document_id.values
subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[45]:


# присваиваем статичные данные
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'views'] = VIEWS_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'depth'] = DEPTH_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'full_reads_percent'] = FPR_UKR

subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[ ]:





# In[46]:


subm.head()


# In[47]:


subm.to_csv(os.path.join(DIR_SUBM, f'{NAME}.csv'), index = False)


# In[ ]:




