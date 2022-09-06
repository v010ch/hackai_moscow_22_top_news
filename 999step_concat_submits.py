#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[ ]:


get_ipython().run_line_magic('watermark', '')


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


get_ipython().run_line_magic('watermark', '--iversions')


# In[ ]:





# In[ ]:





# ## Выствляем переменные

# In[4]:


DIR_DATA   = os.path.join(os.getcwd(), 'data')
DIR_MODELS = os.path.join(os.getcwd(), 'models')
DIR_SUBM   = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_PART = os.path.join(os.getcwd(), 'subm', 'partial')


# In[5]:


# номер попытки и название файлов сабмитов
NTRY = 28
NAME_ENS = f'{NTRY}_ens_pca64_sber_nose_iter_2mod'
NAME_CB  = f'{NTRY}_cb_pca64_sber_bord_nose_iter_2mod'
NAME_XGB = f'{NTRY}_xgb_pca64_sber_bord_nose_iter_2mod'
NAME_LGB = f'{NTRY}_lgb_pca64_sber_bord_nose_iter_2mod'

NAME_MN = f'{NTRY}_mn_pca64_sber_nose_iter_2mod'
#xgb_pca64_sber_lags_parse_bord_nose_val_part


# In[6]:


# константы по Украине для замены в сабмитах 
VIEWS_UKR = 2554204
DEPTH_UKR = 1.799
FPR_UKR = 4.978


# In[ ]:





# ## Блок для воспроизводимости результатов

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





# # Среднее

# In[8]:


test      = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'),  usecols = ['document_id', 'distrib_brdr', 'spec',])
cb_test   = pd.read_csv(os.path.join(DIR_SUBM, f'{NAME_CB}.csv'))
xgb_test  = pd.read_csv(os.path.join(DIR_SUBM, f'{NAME_XGB}.csv'))
lgb_test  = pd.read_csv(os.path.join(DIR_SUBM, f'{NAME_LGB}.csv'))


# In[9]:


print('before ', test.shape, cb_test.shape, lgb_test.shape, xgb_test.shape)
test = pd.concat([test.reset_index(drop = True),
                   cb_test.reset_index(drop = True),
                   xgb_test.reset_index(drop = True),
                   lgb_test.reset_index(drop = True),
                  ], ignore_index = True, axis = 1)
test_cols = ['document_id', 'distrib_brdr', 'spec',
         'document_id_cb', 'views_pred_cb', 'depth_pred_cb', 'frp_pred_cb',
         'document_id_xgb', 'views_pred_xgb', 'depth_pred_xgb', 'frp_pred_xgb',
         'document_id_lgb', 'views_pred_lgb', 'depth_pred_lgb', 'frp_pred_lgb',
        ]
test.columns  = test_cols


if sum(test.document_id == test.document_id_xgb) != test.shape[0] or    sum(test.document_id == test.document_id_lgb) != test.shape[0]:
    print('wtf test')    
    
test.drop(['document_id_cb', 'document_id_xgb', 'document_id_lgb'], axis = 1, inplace = True)
print('after ', test.shape)


# In[10]:


test.columns


# In[1]:


def get_views_mean(inp_row: pd.DataFrame) -> float:
    """
    Подсчет среднего views по строке датафрейма (по сабмитам от трех моделей)
    args:
        inp_row - строка датафрейма по которой считается среднее
    """
    
    #print(inp_row[3], inp_row[6], inp_row[9]) 
    val = np.mean([inp_row[3], inp_row[6], inp_row[9]])
    
    return val


def get_depth_mean(inp_row: pd.DataFrame) -> float:
    """
    Подсчет среднего depth по строке датафрейма (по сабмитам от трех моделей)
    args:
        inp_row - строка датафрейма по которой считается среднее
    """
    #print(inp_row[4], inp_row[7], inp_row[10])
    val = np.mean([inp_row[4], inp_row[7], inp_row[10]])
    
    return val



def get_frp_mean(inp_row: pd.DataFrame) -> float:
    """
    Подсчет среднего frp по строке датафрейма (по сабмитам от трех моделей)
    args:
        inp_row - строка датафрейма по которой считается среднее
    """
    #print(inp_row[5], inp_row[8], inp_row[11])
    val = np.mean([inp_row[5], inp_row[8], inp_row[11]])
    
    return val


# In[12]:


subm_mn = pd.DataFrame()
subm_mn[['document_id']] = test[['document_id']]
#subm = subm.reindex(['document_id', 'distrib_brdr', 'views', 'depth', 'full_reads_percent'], axis = 1)


# In[13]:


subm_mn['views'] = test.apply(get_views_mean, axis = 1)
subm_mn['depth'] = test.apply(get_depth_mean, axis = 1)
subm_mn['full_reads_percent'] = test.apply(get_frp_mean, axis = 1)


# In[14]:


#test.iloc[:5, :].apply(get_views_mean, axis = 1)
#test.iloc[:5, :].apply(get_depth_mean, axis = 1)
#test.iloc[:5, :].apply(get_frp_mean, axis = 1)


# In[15]:


doc_id_ukr = test[test.spec == 1].document_id.values
subm_mn.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[16]:


subm_mn.head()


# In[18]:


#subm.drop(['distrib_brdr'], inplace = True, axis = 1)
subm_mn.to_csv(os.path.join(DIR_SUBM, f'{NAME_MN}.csv'), index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Ансамбли

# In[8]:


def plot_importance(inp_model, imp_number: Optional[int] = 30, imp_type: Optional[str] = 'weight') -> None:
    """
    Функция построения и отображения важности признаков в модели
    args:
        inp_model - модель из которой берется важности признаков
        inp_number - количество признаков для отображения (опционально, 30)
        imp_type - тип по которому определяется важность признака (опционально, 'weight')
    """
    feature_important = inp_model.get_booster().get_score(importance_type = imp_type)
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    data = pd.DataFrame(data = values, index = keys, columns = ["score"]).sort_values(by = "score", ascending = False)
    data.nlargest(imp_number, columns = "score").plot(kind = 'barh', figsize = (30,16))
    


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


def r2(y_pred: np.ndarray, y_true: xgb.DMatrix) -> Tuple[str, float]:
    """
    Функция для расчета дополнительной метрики R2 в cv.
    
    args:
        y_pred - предсказанные значения
        y_true - целевые значения
    return:
        Tuple
            str - имя метрики для отображения в cv scores
            float - значение метрики
    """
    
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


def get_model(inp_df: pd.DataFrame, inp_params: Dict, target: str):
    """
    Обучение модели с выбором оптимального количества итераци по cv на 5 фолдов
    по метрике rmse-mean на валидационных фолдах
    args:
        inp_df - датасет для обучения
        inp_params - параметры модели
        target - имя колонки целевого значения
    return:
        обученная модель XGBRegressor
    """
    
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




