#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[3]:


get_ipython().run_line_magic('watermark', '')


# In[4]:


import time
notebookstart= time.time()


# In[6]:


import os
import pickle as pkl

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn import preprocessing

from typing import Tuple, Dict, List, Optional


# In[7]:


from xgboost import __version__ as xgb_version
from sklearn import __version__ as sklearn_version

print(f'xgb_version: {xgb_version}')
print(f'sklearn_version: {sklearn_version}')


# In[8]:


get_ipython().run_line_magic('watermark', '--iversions')


# In[ ]:





# ## Блок для воспроизводимости результатов

# In[9]:


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





# ## Выствляем переменные

# In[10]:


DIR_DATA   = os.path.join(os.getcwd(), 'data')
DIR_MODELS = os.path.join(os.getcwd(), 'models')
DIR_SUBM   = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_PART = os.path.join(os.getcwd(), 'subm', 'partial')


# In[ ]:





# In[11]:


# номер попытки и название файла сабмита
NTRY = 32
NAME = f'{NTRY}_xgb_pca64_sber_bord_nose_iter_2mod'


# In[12]:


# константы по Украине для замены в сабмитах 
VIEWS_UKR = 2554204
DEPTH_UKR = 1.799
FPR_UKR = 4.978


# In[ ]:





# ## Загрузка данных

# In[13]:


df_train  = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'))#, index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'))#, index_col= 0)
    
with open(os.path.join(DIR_DATA, 'clmns.pkl'), 'rb') as pickle_file:
    clmns = pkl.load(pickle_file)


# In[12]:


df_train.shape, df_test.shape, 


# Распределяем признаки по числовым и категориальным

# In[13]:


cat_cols = []
num_cols = []

for el in clmns.keys():
    cat_cols.extend(clmns[el]['cat'])
    num_cols.extend(clmns[el]['num'])
    if len(clmns[el]['both']) != 0:
        print(clmns[el]['both'])


# In[14]:


num_cols.extend(['hour', 'mounth', 'dow',
                'ph_report', 'ph_gallery', 'tv_prog', 'online', 'video', 'infogr',
                 'holiday', 'day_before_holiday', 'day_after_holiday', #'distrib_brdr',
                 'two_articles',
                 #'spec_event_1',
                ])
#cat_cols.extend(['dow'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


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
    


# In[16]:


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


# In[ ]:


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





# ## views

# In[17]:


#xgb.set_config(verbosity=0)


# In[18]:


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


# In[21]:


get_ipython().run_cell_magic('time', '', "model_views_start = get_model(df_train[df_train.distrib_brdr == 1], xgb_params_views, 'views')")

11     22384.024923      903.611058    61200.728146     19997.9598       0.928265      0.004713       0.45129     0.093598
# In[22]:


plot_importance(model_views_start, 30, 'weight')


# In[ ]:





# In[23]:


get_ipython().run_cell_magic('time', '', "model_views_end = get_model(df_train[df_train.distrib_brdr == 0], xgb_params_views, 'views')")

24      2819.551404      172.393659     9400.983694     715.835259       0.963838      0.004991      0.596266     0.012526
# In[24]:


plot_importance(model_views_end, 30, 'weight')


# In[ ]:





# ## depth

# In[25]:


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


# In[27]:


get_ipython().run_cell_magic('time', '', "model_depth_start = get_model(df_train[df_train.distrib_brdr == 1], xgb_params_depth, 'depth')")

18         0.019287        0.000396        0.035885       0.003551       0.863268      0.011823      0.501045      0.09238
# In[28]:


plot_importance(model_depth_start, 30, 'weight')


# In[ ]:





# In[29]:


get_ipython().run_cell_magic('time', '', "model_depth_end = get_model(df_train[df_train.distrib_brdr == 0], xgb_params_depth, 'depth')")

21         0.009228        0.000468        0.015788       0.002203       0.817579      0.009464      0.460251     0.049625
# In[30]:


plot_importance(model_depth_end, 30, 'weight')


# In[ ]:





# ## full_reads_percent

# In[ ]:





# In[52]:


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


# In[33]:


get_ipython().run_cell_magic('time', '', "model_frp_start = get_model(df_train[df_train.distrib_brdr == 1], xgb_params_frp, 'full_reads_percent')")

13         4.108283          0.1468        7.304048       0.081449       0.847669      0.011282      0.518463     0.020976
# In[34]:


plot_importance(model_frp_start, 30, 'weight')


# In[ ]:





# In[35]:


get_ipython().run_cell_magic('time', '', "model_frp_end = get_model(df_train[df_train.distrib_brdr == 0], xgb_params_frp, 'full_reads_percent')")

13          3.88763        0.069976        6.835213       0.144784       0.838307      0.007139       0.49987     0.014559
# In[36]:


plot_importance(model_frp_end, 30, 'weight')


# In[ ]:





# # Сохраняем предсказания для ансамблей / стекинга

# In[ ]:





# In[37]:


pred_train = pd.DataFrame()
pred_train[['document_id', 'distrib_brdr']] = df_train[['document_id', 'distrib_brdr']]
pred_train = pred_train.reindex(['document_id', 'distrib_brdr', 'views', 'depth', 'full_reads_percent'], axis = 1)


# In[38]:


pred_train.loc[pred_train.query('distrib_brdr == 1').index, 'views'] = model_views_start.predict(df_train[df_train.distrib_brdr == 1][num_cols])
pred_train.loc[pred_train.query('distrib_brdr == 0').index, 'views'] = model_views_end.predict(df_train[df_train.distrib_brdr == 0][num_cols])
print(sum(pred_train.views.isna()), ' Nan in views')


# In[39]:


pred_train.loc[pred_train.query('distrib_brdr == 1').index, 'depth'] = model_depth_start.predict(df_train[df_train.distrib_brdr == 1][num_cols])
pred_train.loc[pred_train.query('distrib_brdr == 0').index, 'depth'] = model_depth_end.predict(df_train[df_train.distrib_brdr == 0][num_cols])
print(sum(pred_train.depth.isna()), ' Nan in depth')


# In[40]:


pred_train.loc[pred_train.query('distrib_brdr == 1').index, 'full_reads_percent'] = model_frp_start.predict(df_train[df_train.distrib_brdr == 1][num_cols])
pred_train.loc[pred_train.query('distrib_brdr == 0').index, 'full_reads_percent'] = model_frp_end.predict(df_train[df_train.distrib_brdr == 0][num_cols])
print(sum(pred_train.full_reads_percent.isna()), ' Nan in full_reads_percent')


# In[41]:


pred_train.drop(['distrib_brdr'], axis =1, inplace = True)
pred_train.to_csv(os.path.join(DIR_SUBM_PART, f'{NAME}_train_part.csv'), index = False)


# In[ ]:





# In[ ]:





# ## Сохранение моделей

# In[42]:


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





# ## Предсказание / сабмит

# In[43]:


subm = pd.DataFrame()
subm[['document_id', 'distrib_brdr']] = df_test[['document_id', 'distrib_brdr']]
subm = subm.reindex(['document_id', 'distrib_brdr', 'views', 'depth', 'full_reads_percent'], axis = 1)


# In[44]:


subm.loc[subm.query('distrib_brdr == 1').index, 'views'] = model_views_start.predict(df_test[df_test.distrib_brdr == 1][num_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'views'] = model_views_end.predict(df_test[df_test.distrib_brdr == 0][num_cols])
print(sum(subm.views.isna()), ' Nan in views')


# In[45]:


subm.loc[subm.query('distrib_brdr == 1').index, 'depth'] = model_depth_start.predict(df_test[df_test.distrib_brdr == 1][num_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'depth'] = model_depth_end.predict(df_test[df_test.distrib_brdr == 0][num_cols])
print(sum(subm.depth.isna()), ' Nan in depth')


# In[46]:


subm.loc[subm.query('distrib_brdr == 1').index, 'full_reads_percent'] = model_frp_start.predict(df_test[df_test.distrib_brdr == 1][num_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'full_reads_percent'] = model_frp_end.predict(df_test[df_test.distrib_brdr == 0][num_cols])
print(sum(subm.full_reads_percent.isna()), ' Nan in full_reads_percent')


# In[ ]:





# Заменяем константные значения по Украине

# In[47]:


doc_id_ukr = df_test[df_test.spec == 1].document_id.values
subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[48]:


# присваиваем статичные данные
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'views'] = VIEWS_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'depth'] = DEPTH_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'full_reads_percent'] = FPR_UKR

subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[49]:


subm.head()


# In[50]:


subm.drop(['distrib_brdr'], inplace = True, axis = 1)
subm.to_csv(os.path.join(DIR_SUBM, f'{NAME}.csv'), index = False)


# In[ ]:





# In[51]:


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


# In[ ]:




