#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[2]:


get_ipython().run_line_magic('watermark', '')


# In[3]:


import os
import pickle as pkl
#import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn import preprocessing
import lightgbm as lgb

from itertools import product
from typing import Tuple


# In[4]:


from tqdm.auto import tqdm
tqdm.pandas()


# In[5]:


#from catboost import __version__ as cb_version
from sklearn import __version__ as sklearn_version

#print(f'cb_version: {cb_version}')
print(f'sklearn_version: {sklearn_version}')


# In[6]:


get_ipython().run_line_magic('watermark', '--iversions')


# In[ ]:





# In[ ]:





# ## Reproducibility block

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

CB_RANDOMSEED  = 309487
XGB_RANDOMSEED = 56
LGB_RANDOMSEED = 874256


# In[ ]:





# In[8]:


DIR_DATA   = os.path.join(os.getcwd(), 'data')
DIR_MODELS = os.path.join(os.getcwd(), 'models')
DIR_SUBM   = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_PART = os.path.join(os.getcwd(), 'subm', 'partial')


# In[ ]:





# In[9]:


NTRY = 25
NAME = f'{NTRY}_lgb_pca64_sber_bord_nose_iter_2mod'


# In[10]:


VIEWS_UKR = 2554204
DEPTH_UKR = 1.799
FPR_UKR = 4.978


# In[ ]:





# ## Load data

# In[11]:


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'))#, index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'))#, index_col= 0)

with open(os.path.join(DIR_DATA, 'clmns.pkl'), 'rb') as pickle_file:
    clmns = pkl.load(pickle_file)


# In[12]:


df_train.shape, df_test.shape,


# In[13]:


df_train['category'] = df_train['category'].astype('category')
df_test['category']  = df_test['category'].astype('category')


# In[14]:


cat_cols = []
num_cols = []

for el in clmns.keys():
    cat_cols.extend(clmns[el]['cat'])
    num_cols.extend(clmns[el]['num'])
    if len(clmns[el]['both']) != 0:
        print(clmns[el]['both'])


# In[15]:


num_cols.extend(['hour', 'mounth'])
cat_cols.extend([ 'dow',
                'ph_report', 'ph_gallery', 'tv_prog', 'online', 'video', 'infogr',
                 'holiday', 'day_before_holiday', 'day_after_holiday', #'distrib_brdr',
                 #'spec_event_1',
                ])


# In[16]:


def r2(preds: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    label = data.get_label()
    #weight = data.get_weight()
    #p_dash = (1 - label) + (2 * label - 1) * preds
    #loss_by_example = - np.log(p_dash)
    #loss = np.average(loss_by_example, weights=weight)

    # # eval_name, eval_result, is_higher_better
    return 'r2', r2_score(preds, label), True


# In[ ]:





# In[ ]:





# In[ ]:




lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    "learning_rate": 0.01,
    "num_leaves": 180,
    "feature_fraction": 0.50,
    "bagging_fraction": 0.50,
    'bagging_freq': 4,
    "max_depth": -1,
    "reg_alpha": 0.3,
    "reg_lambda": 0.1,
    #"min_split_gain":0.2,
    "min_child_weight":10,
    'zero_as_missing':True
                }
# In[17]:


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





# In[ ]:





# In[18]:


df_train[df_train.distrib_brdr == 1].shape, df_train[df_train.distrib_brdr == 0].shape


# In[19]:


df_test[df_test.distrib_brdr == 1].shape, df_test[df_test.distrib_brdr == 0].shape


# In[20]:


#views
train_views_start = lgb.Dataset(df_train[df_train.distrib_brdr == 1][cat_cols + num_cols],
                             df_train[df_train.distrib_brdr == 1][['views']],
                             #feature_name = [cat_cols + num_cols]
                            )
train_views_end = lgb.Dataset(df_train[df_train.distrib_brdr == 0][cat_cols + num_cols],
                             df_train[df_train.distrib_brdr == 0][['views']],
                             #feature_name = [cat_cols + num_cols]
                            )

#lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)


#depth
train_depth_start = lgb.Dataset(df_train[df_train.distrib_brdr == 1][cat_cols + num_cols],
                             df_train[df_train.distrib_brdr == 1][['depth']],
                             #feature_name = [cat_cols + num_cols]
                            )
train_depth_end = lgb.Dataset(df_train[df_train.distrib_brdr == 0][cat_cols + num_cols],
                             df_train[df_train.distrib_brdr == 0][['depth']],
                             #feature_name = [cat_cols + num_cols]
                            )

#full_reads_percent
train_frp_start = lgb.Dataset(df_train[df_train.distrib_brdr == 1][cat_cols + num_cols],
                             df_train[df_train.distrib_brdr == 1][['full_reads_percent']],
                             #feature_name = [cat_cols + num_cols]
                            )
train_frp_end = lgb.Dataset(df_train[df_train.distrib_brdr == 0][cat_cols + num_cols],
                             df_train[df_train.distrib_brdr == 0][['full_reads_percent']],
                             #feature_name = [cat_cols + num_cols]
                            )


# In[ ]:





# In[ ]:




gbdt, traditional Gradient Boosting Decision Tree, aliases: gbrt
rf, Random Forest, aliases: random_forest
dart, Dropouts meet Multiple Additive Regression Trees
goss, Gradient-based One-Side Sampling
# ## views

# In[21]:


# defining parameters 
params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    #'num_leaves': 6,
    #'learning_rate': 0.05,
    #'metric': {'l2','l1'},
    'metric': {'rmse'},
    'verbose': -1,
    
    #'reg_alpha': 5,   # != 0  Hard L1 regularization
    #'reg_lambda': 10,   # != 0  Hard L2 regularization
    
    #'lambda_l1': 5, 
    #'lambda_l2': 5,
    
    'random_seed': LGB_RANDOMSEED,
}


# In[ ]:





# In[22]:


def get_model(inp_ds, inp_params):
    
    score = lgb.cv(inp_params, 
                 inp_ds, 
                 #num_boost_round = 10000,
                 num_boost_round=600,
                 nfold = 5,
                 verbose_eval = 500,
                 #early_stopping_rounds = 100,
                 stratified = False,
                 eval_train_metric = r2,
                 feval = r2,
                 #return_cvbooster = True,
                )
    print(np.argmin(score['valid rmse-mean']), score['train rmse-mean'][np.argmin(score['valid rmse-mean'])], score['train rmse-stdv'][np.argmin(score['valid rmse-mean'])], )
    print(np.argmin(score['valid rmse-mean']), score['valid rmse-mean'][np.argmin(score['valid rmse-mean'])], score['valid rmse-stdv'][np.argmin(score['valid rmse-mean'])], )

    print(np.argmax(score['valid r2-mean']), score['train r2-mean'][np.argmax(score['valid r2-mean'])], score['train r2-stdv'][np.argmax(score['valid r2-mean'])], )
    print(np.argmax(score['valid r2-mean']), score['valid r2-mean'][np.argmax(score['valid r2-mean'])], score['valid r2-stdv'][np.argmax(score['valid r2-mean'])], )
    
    
    niters = np.argmin(score['valid rmse-mean'])
    print(niters)
    
    lgb_model = lgb.train(inp_params,
                            train_set = inp_ds,
                            num_boost_round = niters,
                            #early_stopping_rounds=30,
                            verbose_eval = False,
                           )
    
    
    return lgb_model


# In[ ]:




%%time
score_v_s = lgb.cv(params, 
                 train_views_full_start, 
                 #num_boost_round = 10000,
                 num_boost_round=600,
                 nfold = 5,
                 verbose_eval = 500,
                 #early_stopping_rounds = 100,
                 stratified = False,
                 eval_train_metric = r2,
                 feval = r2,
                 #return_cvbooster = True,
                )
print(np.argmin(score_v_s['valid rmse-mean']), score_v_s['train rmse-mean'][np.argmin(score_v_s['valid rmse-mean'])], score_v_s['train rmse-stdv'][np.argmin(score_v_s['valid rmse-mean'])], )
print(np.argmin(score_v_s['valid rmse-mean']), score_v_s['valid rmse-mean'][np.argmin(score_v_s['valid rmse-mean'])], score_v_s['valid rmse-stdv'][np.argmin(score_v_s['valid rmse-mean'])], )

print(np.argmax(score_v_s['valid r2-mean']), score_v_s['train r2-mean'][np.argmax(score_v_s['valid r2-mean'])], score_v_s['train r2-stdv'][np.argmax(score_v_s['valid r2-mean'])], )
print(np.argmax(score_v_s['valid r2-mean']), score_v_s['valid r2-mean'][np.argmax(score_v_s['valid r2-mean'])], score_v_s['valid r2-stdv'][np.argmax(score_v_s['valid r2-mean'])], )
# In[23]:


#score_v.keys()


# In[24]:


#if np.argmin(score_v['valid rmse-mean']) != np.argmax(score_v['valid r2-mean']):
#    raise ValueError('wtf?', np.argmin(score_v['valid rmse-mean']), np.argmax(score_v['valid r2-mean']))

1661 48897.91696678655 9179.139175918961#views_iter = np.argmax(score_v['valid r2-mean'])
views_start_iter = np.argmin(score_v['valid rmse-mean'])
print(views_iter)lgb_model_views = lgb.train(params,
                            train_set=train_views_full_start,
                            num_boost_round = views_iter,
                            #early_stopping_rounds=30,
                            verbose_eval = False,
                           )
# In[25]:


get_ipython().run_cell_magic('time', '', 'lgb_model_views_start = get_model(train_views_start, params)')

226 17349.160056309396 5517.927217157143
226 57506.143468314796 19351.41192630733
# In[26]:


lgb.plot_importance(lgb_model_views_start, max_num_features = 30, figsize = (30, 16), importance_type = 'gain')
#lgb.plot_importance(lgb_model_views, max_num_features = 30, figsize = (30, 16), importance_type = 'split')
# importance_type (str, optional (default="auto")) – How the importance is calculated. If “auto”, if booster parameter is LGBMModel, booster.importance_type attribute is used; 
# “split” otherwise. If “split”, result contains numbers of times the feature is used in a model. If “gain”, result contains total gains of splits which use the feature.


# In[ ]:





# In[ ]:





# In[27]:


get_ipython().run_cell_magic('time', '', 'lgb_model_views_end = get_model(train_views_end, params)')

77 3761.9464700862563 110.23969646181965
77 9034.796767591883 653.1865103504509
# In[28]:


lgb.plot_importance(lgb_model_views_start, max_num_features = 30, figsize = (30, 16), importance_type = 'gain')


# In[ ]:





# In[ ]:





# In[ ]:





# ## depth

# In[29]:


# defining parameters 
params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    #'num_leaves': 10,
    #'learning_rate': 0.05,
    #'metric': {'l2','l1'},
    'metric': {'rmse'},
    'verbose': -1,
    
    #'reg_alpha': 5,   # != 0  Hard L1 regularization
    #'reg_lambda': 10,   # != 0  Hard L2 regularization
    
    #'lambda_l1': 5, 
    #'lambda_l2': 5,
    
    'random_seed': LGB_RANDOMSEED,
}

%%time
score_d = lgb.cv(params, 
                 train_depth_full, 
                 #num_boost_round = 10000,
                 num_boost_round=600,
                 nfold = 5,
                 verbose_eval = 500,
                 #early_stopping_rounds = 100,
                 stratified = False,
                 eval_train_metric = r2,
                 feval = r2,
                 #return_cvbooster = True,
                )
print(np.argmin(score_d['valid rmse-mean']), score_d['train rmse-mean'][np.argmin(score_d['valid rmse-mean'])], score_d['train rmse-stdv'][np.argmin(score_d['valid rmse-mean'])], )
print(np.argmin(score_d['valid rmse-mean']), score_d['valid rmse-mean'][np.argmin(score_d['valid rmse-mean'])], score_d['valid rmse-stdv'][np.argmin(score_d['valid rmse-mean'])], )

print(np.argmax(score_d['valid r2-mean']), score_d['train r2-mean'][np.argmax(score_d['valid r2-mean'])], score_d['train r2-stdv'][np.argmax(score_d['valid r2-mean'])], )
print(np.argmax(score_d['valid r2-mean']), score_d['valid r2-mean'][np.argmax(score_d['valid r2-mean'])], score_d['valid r2-stdv'][np.argmax(score_d['valid r2-mean'])], )378 0.02735314197033909 0.0009451007887406206
# In[30]:


#if np.argmin(score_d['valid rmse-mean']) != np.argmax(score_d['valid r2-mean']):
#    raise ValueError('wtf?', np.argmin(score_d['valid rmse-mean']), np.argmax(score_d['valid r2-mean']))

#depth_iter = np.argmax(score_d['valid r2-mean'])
depth_iter = np.argmin(score_d['valid rmse-mean'])
print(depth_iter)# fitting the model
lgb_model_depth = lgb.train(params,
                            train_set=train_depth_full,
                            num_boost_round = depth_iter,
                            verbose_eval = False,
                           )
# In[ ]:





# In[31]:


get_ipython().run_cell_magic('time', '', 'lgb_model_depth_start = get_model(train_depth_start, params)')

92 0.012774675718467876 0.00043797150418636567
92 0.03574606040187936 0.0030533996509353564
# In[32]:


lgb.plot_importance(lgb_model_depth_start, max_num_features = 30, figsize = (30, 16), importance_type = 'gain')
#lgb.plot_importance(lgb_model_depth, max_num_features = 30, figsize = (30, 16), importance_type = 'split')


# In[ ]:





# In[ ]:





# In[33]:


get_ipython().run_cell_magic('time', '', 'lgb_model_depth_end = get_model(train_depth_end, params)')

48 0.010399829750017912 0.0008243396934630533
48 0.015475994859493908 0.002194558006310076
# In[34]:


lgb.plot_importance(lgb_model_depth_end, max_num_features = 30, figsize = (30, 16), importance_type = 'gain')
#lgb.plot_importance(lgb_model_depth, max_num_features = 30, figsize = (30, 16), importance_type = 'split')


# In[ ]:





# ## full_reads_percent
score_f = lgb.cv(params, 
                 train_frp_full, 
                 #num_boost_round = 10000,
                 num_boost_round=600,
                 nfold = 5,
                 verbose_eval = 500,
                 #early_stopping_rounds = 100,
                 stratified = False,
                 eval_train_metric = r2,
                 feval = r2,
                 #return_cvbooster = True,
                )
print(np.argmin(score_f['valid rmse-mean']), score_f['train rmse-mean'][np.argmin(score_f['valid rmse-mean'])], score_f['train rmse-stdv'][np.argmin(score_f['valid rmse-mean'])], )
print(np.argmin(score_f['valid rmse-mean']), score_f['valid rmse-mean'][np.argmin(score_f['valid rmse-mean'])], score_f['valid rmse-stdv'][np.argmin(score_f['valid rmse-mean'])], )

print(np.argmax(score_f['valid r2-mean']), score_f['train r2-mean'][np.argmax(score_f['valid r2-mean'])], score_f['train r2-stdv'][np.argmax(score_f['valid r2-mean'])], )
print(np.argmax(score_f['valid r2-mean']), score_f['valid r2-mean'][np.argmax(score_f['valid r2-mean'])], score_f['valid r2-stdv'][np.argmax(score_f['valid r2-mean'])], )163 7.02038356002961 0.1360496068604094   / 116 / 111
# In[35]:


#if np.argmin(score_f['valid rmse-mean']) != np.argmax(score_f['valid r2-mean']):
#    raise ValueError('wtf?', np.argmin(score_f['valid rmse-mean']), np.argmax(score_f['valid r2-mean']))

#frp_iter = np.argmax(score_f['valid r2-mean'])
frp_iter = np.argmin(score_f['valid rmse-mean'])
print(frp_iter)
# In[36]:


# defining parameters 
params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    #'num_leaves': 10,
    #'learning_rate': 0.05,
    #'metric': {'l2','l1'},
    'metric': {'rmse'},
    'verbose': -1,
    
    #'reg_alpha': 5,   # != 0  Hard L1 regularization
    #'reg_lambda': 10,   # != 0  Hard L2 regularization
    
    #'lambda_l1': 5, 
    #'lambda_l2': 5,
    
    'random_seed': LGB_RANDOMSEED,
}

# fitting the model
lgb_model_frp = lgb.train(params,
                            train_set=train_frp_full,#train_set=train_ds_frp,
                            #valid_sets=val_ds_frp,
                            num_boost_round = frp_iter,
                            #early_stopping_rounds=30,
                            verbose_eval = False,
                           )
# In[37]:


get_ipython().run_cell_magic('time', '', 'lgb_model_frp_start = get_model(train_frp_start, params)')

54 3.5817026929454285 0.015001174048343903
54 7.035757899813923 0.08849771756751056
# In[38]:


lgb.plot_importance(lgb_model_frp_start, max_num_features = 30, figsize = (30, 16), importance_type = 'gain')
#lgb.plot_importance(lgb_model_frp, max_num_features = 30, figsize = (30, 16), importance_type = 'split')


# In[ ]:





# In[ ]:





# In[39]:


get_ipython().run_cell_magic('time', '', 'lgb_model_frp_end = get_model(train_frp_end, params)')

71 3.1057100118726253 0.014663128279285759
71 6.440356846362912 0.14059759791727314
# In[40]:


lgb.plot_importance(lgb_model_frp_start, max_num_features = 30, figsize = (30, 16), importance_type = 'gain')
#lgb.plot_importance(lgb_model_frp, max_num_features = 30, figsize = (30, 16), importance_type = 'split')


# In[ ]:





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

x_train_pred.columns = ['document_id', 'views_pred_lgb', 'depth_pred_lgb', 'frp_pred_lgb']
x_val_pred.columns   = ['document_id', 'views_pred_lgb', 'depth_pred_lgb', 'frp_pred_lgb']

print('after ', x_train_pred.shape)
print('after ', x_val_pred.shape)

x_train_pred.to_csv(os.path.join(DIR_SUBM_PART, f'{NAME}_train_part.csv'), index = False)
x_val_pred.to_csv(os.path.join(DIR_SUBM_PART, f'{NAME}_val_part.csv'), index = False)
# In[ ]:





# ## save models

# In[41]:


lgb_model_views_start.save_model(os.path.join(DIR_MODELS, f'{NAME}_v_start.txt'), num_iteration = lgb_model_views_start.best_iteration)
lgb_model_views_end.save_model(  os.path.join(DIR_MODELS, f'{NAME}_v_end.txt'),   num_iteration = lgb_model_views_end.best_iteration)
lgb_model_depth_start.save_model(os.path.join(DIR_MODELS, f'{NAME}_d_start.txt'), num_iteration = lgb_model_depth_start.best_iteration)
lgb_model_depth_end.save_model(  os.path.join(DIR_MODELS, f'{NAME}_d_end.txt'),   num_iteration = lgb_model_depth_end.best_iteration)
lgb_model_frp_start.save_model(  os.path.join(DIR_MODELS, f'{NAME}_f_start.txt'), num_iteration = lgb_model_frp_start.best_iteration)
lgb_model_frp_end.save_model(    os.path.join(DIR_MODELS, f'{NAME}_f_end.txt'),   num_iteration = lgb_model_frp_end.best_iteration)


# In[ ]:





# ## Делаем предсказание трейна для ансамблей

# In[59]:


pred_train = pd.DataFrame()
pred_train[['document_id', 'distrib_brdr']] = df_train[['document_id', 'distrib_brdr']]
pred_train = pred_train.reindex(['document_id', 'distrib_brdr', 'views', 'depth', 'full_reads_percent'], axis = 1)


# In[60]:


pred_train.loc[pred_train.query('distrib_brdr == 1').index, 'views'] = lgb_model_views_start.predict(df_train[df_train.distrib_brdr == 1][cat_cols + num_cols])
pred_train.loc[pred_train.query('distrib_brdr == 0').index, 'views'] = lgb_model_views_end.predict(df_train[df_train.distrib_brdr == 0][cat_cols + num_cols])
print(sum(pred_train.views.isna()), ' Nan in views')


# In[63]:


pred_train.loc[pred_train.query('distrib_brdr == 1').index, 'depth'] = lgb_model_depth_start.predict(df_train[df_train.distrib_brdr == 1][cat_cols + num_cols])
pred_train.loc[pred_train.query('distrib_brdr == 0').index, 'depth'] = lgb_model_depth_end.predict(df_train[df_train.distrib_brdr == 0][cat_cols + num_cols])
print(sum(pred_train.depth.isna()), ' Nan in depth')


# In[64]:


pred_train.loc[pred_train.query('distrib_brdr == 1').index, 'full_reads_percent'] = lgb_model_frp_start.predict(df_train[df_train.distrib_brdr == 1][cat_cols + num_cols])
pred_train.loc[pred_train.query('distrib_brdr == 0').index, 'full_reads_percent'] = lgb_model_frp_end.predict(df_train[df_train.distrib_brdr == 0][cat_cols + num_cols])
print(sum(pred_train.full_reads_percent.isna()), ' Nan in full_reads_percent')


# In[65]:


pred_train.drop(['distrib_brdr'], axis =1, inplace = True)
pred_train.to_csv(os.path.join(DIR_SUBM_PART, f'{NAME}_train_part.csv'), index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Делаем предсказание для теста
pred_views = lgb_model_views.predict(df_test[cat_cols + num_cols])
pred_depth = lgb_model_depth.predict(df_test[cat_cols + num_cols])
pred_frp   = lgb_model_frp.predict(  df_test[cat_cols + num_cols])subm = pd.DataFrame()
subm['document_id'] = df_test.document_id

subm['views'] = pred_views
subm['depth'] = pred_depth
subm['full_reads_percent'] = pred_frp
# In[47]:


subm = pd.DataFrame()
subm[['document_id', 'distrib_brdr']] = df_test[['document_id', 'distrib_brdr']]
subm = subm.reindex(['document_id', 'distrib_brdr', 'views', 'depth', 'full_reads_percent'], axis = 1)


# In[48]:


subm.loc[subm.query('distrib_brdr == 1').index, 'views'] = lgb_model_views_start.predict(df_test[df_test.distrib_brdr == 1][cat_cols + num_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'views'] = lgb_model_views_end.predict(df_test[df_test.distrib_brdr == 0][cat_cols + num_cols])
sum(subm.views.isna())


# In[49]:


subm.loc[subm.query('distrib_brdr == 1').index, 'depth'] = lgb_model_depth_start.predict(df_test[df_test.distrib_brdr == 1][cat_cols + num_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'depth'] = lgb_model_depth_end.predict(df_test[df_test.distrib_brdr == 0][cat_cols + num_cols])
sum(subm.depth.isna())


# In[50]:


subm.loc[subm.query('distrib_brdr == 1').index, 'full_reads_percent'] = lgb_model_frp_start.predict(df_test[df_test.distrib_brdr == 1][cat_cols + num_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'full_reads_percent'] = lgb_model_frp_end.predict(df_test[df_test.distrib_brdr == 0][cat_cols + num_cols])
sum(subm.full_reads_percent.isna())


# In[51]:


subm.drop(['distrib_brdr'], axis = 1, inplace = True)


# In[ ]:





# In[52]:


doc_id_ukr = df_test[df_test.spec == 1].document_id.values
subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[53]:


# присваиваем статичные данные
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'views'] = VIEWS_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'depth'] = DEPTH_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'full_reads_percent'] = FPR_UKR

subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[54]:


subm.head()


# In[55]:


subm.to_csv(os.path.join(DIR_SUBM, f'{NAME}.csv'), index = False)


# In[ ]:





# In[ ]:




