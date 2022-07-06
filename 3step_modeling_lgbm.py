#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[ ]:


get_ipython().run_line_magic('watermark', '')


# In[ ]:


import os
import pickle as pkl
#import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
import lightgbm as lgb

from itertools import product


# In[ ]:


from tqdm.auto import tqdm
tqdm.pandas()


# In[ ]:


#from catboost import __version__ as cb_version
from sklearn import __version__ as sklearn_version

#print(f'cb_version: {cb_version}')
print(f'sklearn_version: {sklearn_version}')


# In[ ]:


get_ipython().run_line_magic('watermark', '--iversions')


# In[ ]:





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

CB_RANDOMSEED  = 309487
XGB_RANDOMSEED = 56
LGB_RANDOMSEED = 874256


# In[ ]:





# In[ ]:


DIR_DATA   = os.path.join(os.getcwd(), 'data')
DIR_MODELS = os.path.join(os.getcwd(), 'models')
DIR_SUBM   = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_PART = os.path.join(os.getcwd(), 'subm', 'partial')


# In[ ]:





# ## Load data

# In[ ]:


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'))#, index_col= 0)
x_train  = pd.read_csv(os.path.join(DIR_DATA, 'x_train.csv'))#, index_col= 0)
x_val    = pd.read_csv(os.path.join(DIR_DATA, 'x_val.csv'))#, index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'))#, index_col= 0)

with open(os.path.join(DIR_DATA, 'cat_columns.pkl'), 'rb') as pickle_file:
    cat_cols = pkl.load(pickle_file)
    
with open(os.path.join(DIR_DATA, 'num_columns.pkl'), 'rb') as pickle_file:
    num_cols = pkl.load(pickle_file)


# In[ ]:


df_train.shape, df_test.shape, len(cat_cols), len(num_cols), #x_train.shape, x_val.shape,


# отделяем метки от данных
y_train = df_train[['views', 'depth', 'full_reads_percent']]
df_train.drop(['views', 'depth', 'full_reads_percent'], axis = 1, inplace = True)

df_train.shape, y_train.shape
# In[ ]:


y_train = x_train[['views', 'depth', 'full_reads_percent']]
x_train.drop(['views', 'depth', 'full_reads_percent'], axis = 1, inplace = True)

y_val = x_val[['views', 'depth', 'full_reads_percent']]
x_val.drop(['views', 'depth', 'full_reads_percent'], axis = 1, inplace = True)


# In[ ]:


#cat_cols + num_cols


# In[ ]:


cat_cols = cat_cols + ['category']


# In[ ]:


df_train['category'] = df_train['category'].astype('category')
x_train['category']  = x_train['category'].astype('category')
x_val['category']    = x_val['category'].astype('category')
df_test['category']  = df_test['category'].astype('category')


# In[ ]:





# In[ ]:





# In[ ]:


def train_lgb_cat(inp_df, inp_vals, inp_category, inp_cat_cols, inp_num_cols):

    
    num_of_leaves_vars    = [4, 8, 16, 32, 64, 128]
    max_depth_vars        = [4, 8, 16, 32]#, 64, 128]
    min_data_in_leaf_vars = [4, 8, 16, 32]#, 64, 128]
    learn_rate_vars       = [0.1, 0.05, 0.01] #1, 0.5, 
    
    min_rmse = 1000000
    ret_progress = []
    
    for nl, lr, md, mdlf in tqdm(product(num_of_leaves_vars, learn_rate_vars, max_depth_vars, min_data_in_leaf_vars), 
                                total = len(num_of_leaves_vars)*len(learn_rate_vars)*len(max_depth_vars)*len(min_data_in_leaf_vars)
                                ):
    
        params = {
            'task': 'train', 
            'boosting': 'gbdt',
            'objective': 'regression',
            'num_leaves': nl,
            'learning_rate': lr,
            'max_depth': md,
            'min_data_in_leaf': mdlf,
            'metric': {'rmse'},
            #'verbose': -1,
            'random_seed': LGB_RANDOMSEED,
            'force_col_wise': False,
            'n_jobs' : -1,
            
            
            #'reg_alpha': 10,   # != 0  Hard L1 regularization
            #'reg_lambda': 0,   # != 0  Hard L2 regularization
        }


        train_ds_views = lgb.Dataset(inp_df[inp_df.category == inp_category][cat_cols + num_cols],
                                     #inp_df[cat_cols + num_cols],
                                     inp_vals[inp_df.category == inp_category][['views']],
                                     #feature_name = [cat_cols + num_cols]
                                    )

        results = lgb.cv(params, 
                         train_ds_views, 
                         num_boost_round = 10000,
                         nfold = 5,
                         verbose_eval = 500,
                         early_stopping_rounds = 100,
                         stratified = False,
                         #return_cvbooster = True,
                        )

        optimal_rounds = np.argmin(results['rmse-mean'])
        best_cv_score  = min(results['rmse-mean'])

        if best_cv_score < min_rmse:
            ret_progress.append(f'nl={nl:3d}, lr={lr:3f}, md={md:3d}, mdlf={mdlf:3d}, {optimal_rounds}, {best_cv_score}')
        
        #print(nl, lr, md, mdlf, optimal_rounds, best_cv_score)
        print(f'nl={nl:3d}, lr={lr:3f}, md={md:3d}, mdlf={mdlf:3d}, {optimal_rounds}, {best_cv_score}')  
    
    return ret_progress

'5409f11ce063da9c8b588a12', '5409f11ce063da9c8b588a13', '5409f11ce063da9c8b588a18', '540d5eafcbb20f2524fc0509', '540d5ecacbb20f2524fc050a', '5433e5decbb20f277b20eca9'
# In[ ]:


#progress = train_lgb_cat(df_train, y_train, '5409f11ce063da9c8b588a12', cat_cols, num_cols)


# In[ ]:





# In[ ]:





# In[ ]:


#with open(os.path.join(DIR_DATA, 'progress.pkl'), 'wb') as pickle_file:
#    pkl.dump(progress, pickle_file)


# In[ ]:





# In[ ]:





# In[ ]:


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
#!!!!!!!!!!!!!!!!!!!!!!! #lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)


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

# In[ ]:


# defining parameters 
params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 6,
    'learning_rate': 0.05,
    'metric': {'l2','l1'},
    'verbose': -1,
    
    #'reg_alpha': 5,   # != 0  Hard L1 regularization
    'reg_lambda': 10,   # != 0  Hard L2 regularization
    
    'random_seed': LGB_RANDOMSEED,
}


# In[ ]:


# fitting the model
lgb_model_views = lgb.train(params,
                            train_set=train_ds_views,
                            valid_sets=val_ds_views,
                            early_stopping_rounds=30,
                            verbose_eval = False,
                           )


# In[ ]:


# Get predictions and metrics
preds_train_views = lgb_model_views.predict(x_train[cat_cols + num_cols])
preds_val_views   = lgb_model_views.predict(x_val[cat_cols + num_cols])

train_score_views = r2_score(y_train["views"], preds_train_views)
val_score_views   = r2_score(y_val["views"],   preds_val_views)

train_score_views, val_score_views

(0.598891443106734, 0.6282297125590693)
# In[ ]:





# In[ ]:


lgb.plot_importance(lgb_model_views, max_num_features = 30, figsize = (30, 16), importance_type = 'gain')
#lgb.plot_importance(lgb_model_views, max_num_features = 30, figsize = (30, 16), importance_type = 'split')
# importance_type (str, optional (default="auto")) – How the importance is calculated. If “auto”, if booster parameter is LGBMModel, booster.importance_type attribute is used; 
# “split” otherwise. If “split”, result contains numbers of times the feature is used in a model. If “gain”, result contains total gains of splits which use the feature.


# In[ ]:





# In[ ]:





# ## depth

# In[ ]:


# defining parameters 
params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    #'learning_rate': 0.05,
    'metric': {'l2','l1'},
    'verbose': -1,
    
    #'reg_alpha': 5,   # != 0  Hard L1 regularization
    'reg_lambda': 10,   # != 0  Hard L2 regularization
    
    'random_seed': LGB_RANDOMSEED,
}


# In[ ]:


# fitting the model
lgb_model_depth = lgb.train(params,
                            train_set=train_ds_depth,
                            valid_sets=val_ds_depth,
                            early_stopping_rounds=30,
                            verbose_eval = False,
                           )


# In[ ]:


# Get predictions and metrics
preds_train_depth = lgb_model_depth.predict(x_train[cat_cols + num_cols])
preds_val_depth   = lgb_model_depth.predict(x_val[cat_cols + num_cols])

train_score_depth = r2_score(y_train["depth"], preds_train_depth)
val_score_depth   = r2_score(y_val["depth"],   preds_val_depth)

train_score_depth, val_score_depth

(0.8835231811710682, 0.7519392810902077) emb + lag
# In[ ]:





# In[ ]:


lgb.plot_importance(lgb_model_depth, max_num_features = 30, figsize = (30, 16), importance_type = 'gain')
#lgb.plot_importance(lgb_model_depth, max_num_features = 30, figsize = (30, 16), importance_type = 'split')


# In[ ]:





# ## full_reads_percent

# In[ ]:


# defining parameters 
params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    #'learning_rate': 0.05,
    'metric': {'l2','l1'},
    'verbose': -1,
    
    #'reg_alpha': 5,   # != 0  Hard L1 regularization
    'reg_lambda': 10,   # != 0  Hard L2 regularization
    
    'random_seed': LGB_RANDOMSEED,
}


# In[ ]:


# fitting the model
lgb_model_frp = lgb.train(params,
                            train_set=train_ds_frp,
                            valid_sets=val_ds_frp,
                            early_stopping_rounds=30,
                            verbose_eval = False,
                           )


# In[ ]:


# Get predictions and metrics
preds_train_frp = lgb_model_frp.predict(x_train[cat_cols + num_cols])
preds_val_frp  = lgb_model_frp.predict(x_val[cat_cols + num_cols])

train_score_frp = r2_score(y_train["full_reads_percent"], preds_train_frp)
val_score_frp  = r2_score(y_val["full_reads_percent"],   preds_val_frp)

train_score_frp, val_score_frp

(0.5942299501576211, 0.3778378916210692) emb + lag
# In[ ]:





# In[ ]:


lgb.plot_importance(lgb_model_frp, max_num_features = 30, figsize = (30, 16), importance_type = 'gain')
#lgb.plot_importance(lgb_model_frp, max_num_features = 30, figsize = (30, 16), importance_type = 'split')


# In[ ]:





# In[ ]:





# In[ ]:


score_train = 0.4 * train_score_views + 0.3 * train_score_depth + 0.3 * train_score_frp
score_val  = 0.4 * val_score_views  + 0.3 * val_score_depth  + 0.3 * val_score_frp

score_train, score_val

(0.5476311546688098, 0.5221479268702258)
# In[ ]:





# In[ ]:


NTRY = 5


# ## save models

# In[ ]:


lgb_model_views.save_model(os.path.join(DIR_MODELS, f'{NTRY}_lgm_views.txt'), num_iteration = lgb_model_views.best_iteration)
lgb_model_depth.save_model(os.path.join(DIR_MODELS, f'{NTRY}_lgm_depth.txt'), num_iteration = lgb_model_depth.best_iteration)
lgb_model_frp.save_model(  os.path.join(DIR_MODELS, f'{NTRY}_lgm_frp.txt'),   num_iteration = lgb_model_frp.best_iteration)


# In[ ]:





# ## make predict

# In[ ]:


pred_views = lgb_model_views.predict(df_test[cat_cols + num_cols])
pred_depth = lgb_model_depth.predict(df_test[cat_cols + num_cols])
pred_frp   = lgb_model_frp.predict(  df_test[cat_cols + num_cols])


# In[ ]:


subm = pd.DataFrame()
subm['document_id'] = df_test.document_id

subm['views'] = pred_views
subm['depth'] = pred_depth
subm['full_reads_percent'] = pred_frp


# In[ ]:


subm.head()


# In[ ]:


subm.to_csv(os.path.join(DIR_SUBM, f'{NTRY}_lgb_ttl_emb_depth_frp.csv'), index = False)


# In[ ]:




