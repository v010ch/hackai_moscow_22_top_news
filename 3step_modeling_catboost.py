#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[2]:


get_ipython().run_line_magic('watermark', '')


# In[3]:


import time
notebookstart= time.time()


# In[4]:


import os
import pickle as pkl

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn import preprocessing
from catboost import CatBoostRegressor, Pool, cv

import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


from catboost import __version__ as cb_version
from sklearn import __version__ as sklearn_version

print(f'cb_version: {cb_version}')
print(f'sklearn_version: {sklearn_version}')


# In[6]:


get_ipython().run_line_magic('watermark', '--iversions')


# In[ ]:





# In[ ]:





# ## Блок для воспроизводимости решений

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


NTRY = 32
NAME = f'{NTRY}_cb_pca64_sber_bord_nose_iter_2mod'


# In[10]:


VIEWS_UKR = 2554204
DEPTH_UKR = 1.799
FPR_UKR = 4.978


# In[ ]:





# ## Загружаем данные

# In[11]:


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'), index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'), index_col= 0)

with open(os.path.join(DIR_DATA, 'clmns.pkl'), 'rb') as pickle_file:
    clmns = pkl.load(pickle_file)


# In[12]:


df_train.shape, df_test.shape, 


# In[ ]:





# Формируем списки числовых и категориальных переменных.   
# Полностью категориальные и полностью числовых формируются автоматически.   

# In[13]:


cat_cols = []
num_cols = []

for el in clmns.keys():
    cat_cols.extend(clmns[el]['cat'])
    num_cols.extend(clmns[el]['num'])
    if len(clmns[el]['both']) != 0:
        print(clmns[el]['both'])


# Остается распределить признаки, которые могу быть как категориальными, так и числовыми.    
# Сделаем это.

# In[14]:


num_cols.extend(['hour', 'mounth'])
cat_cols.extend(['dow', 
                 'ph_report', 'ph_gallery', 'tv_prog', 'online', 'video', 'infogr',
                 'holiday', 'day_before_holiday', 'day_after_holiday', #'distrib_brdr',
                 #'spec_event_1',
                ])


# In[15]:


for el in cat_cols:
    df_train[el] = df_train[el].astype(str)
    df_test[el] = df_test[el].astype(str)


# In[ ]:





# Создаем Catboost Pools для обучения с условием разделения на start (данные до 2022-04-08) / end (данные после 2022-04-08)

# In[16]:


#views
train_views_start = Pool(df_train[df_train.distrib_brdr == 1][cat_cols + num_cols],
                         df_train[df_train.distrib_brdr == 1][['views']],
                         cat_features = cat_cols,
                        )
train_views_end = Pool(df_train[df_train.distrib_brdr == 0][cat_cols + num_cols],
                       df_train[df_train.distrib_brdr == 0][['views']],
                       cat_features = cat_cols,
                      )
#depth
train_depth_start = Pool(df_train[df_train.distrib_brdr == 1][cat_cols + num_cols],
                         df_train[df_train.distrib_brdr == 1][['depth']],
                         cat_features = cat_cols,
                        )
train_depth_end = Pool(df_train[df_train.distrib_brdr == 0][cat_cols + num_cols],
                       df_train[df_train.distrib_brdr == 0][['depth']],
                       cat_features = cat_cols,
                      )

#frp
train_frp_start = Pool(df_train[df_train.distrib_brdr == 1][cat_cols + num_cols],
                       df_train[df_train.distrib_brdr == 1][['full_reads_percent']],
                       cat_features = cat_cols,
                      )
train_frp_end = Pool(df_train[df_train.distrib_brdr == 0][cat_cols + num_cols],
                     df_train[df_train.distrib_brdr == 0][['full_reads_percent']],
                     cat_features = cat_cols,

                    )


# In[ ]:





# In[17]:


def plot_feature_importance2(inp_model, inp_pool, imp_number = 30):
    
    data = pd.DataFrame({'feature_importance': inp_model.get_feature_importance(inp_pool), 
              'feature_names': inp_pool.get_feature_names()}).sort_values(by=['feature_importance'], 
                                                       ascending=True)
    
    data.nlargest(imp_number, columns="feature_importance").plot(kind='barh', figsize = (30,16)) ## plot top 40 features


# In[18]:


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

# Обучаем модели для views с разделением на модели start (данные до 2022-04-08) / end (данные после 2022-04-08) 

# In[19]:


cb_params_views = {"iterations": 2500,
                  #"depth": 2,
                  "loss_function": "RMSE",
                  'eval_metric': 'R2',
                  "verbose": False
                  }


# Рассчитываем на cv оптимальное количество итераций по средней R2 на валидационных фолдах 

# In[20]:


def get_model(inp_pool, inp_params):
    
    
    scores = cv(inp_pool,
                      inp_params,
                      fold_count=5,
                      seed = CB_RANDOMSEED, 
                      #plot="True"
                     )
    
    # проверка что лучшие итерации по test-RMSE-mean и test-R2-mean одинаковы
    # для cb чаще правда, для lgb и xgb чаще ложь
    #if scores['test-RMSE-mean'].argmin() != scores['test-R2-mean'].argmax():
    #    raise ValueError('wtf?', scores['test-RMSE-mean'].argmin(), scores['test-R2-mean'].argmax())
    
    print(scores[scores['test-R2-mean'] == scores['test-R2-mean'].max()].to_string())
    
    # из cv берем лучшее по test-R2-mean количество итераций
    # на нем и обучаем моедль на всех данных
    niters = scores['test-R2-mean'].argmax()
    print(niters)
    
    cb_model = CatBoostRegressor(iterations = niters,
                                 #learning_rate=0.05,
                                 #depth=10,
                                 random_seed = CB_RANDOMSEED,
                                 #n_estimators=100,
                                )
    # Fit model
    cb_model.fit(inp_pool,
                 #plot = True,
                 verbose = 100,
                )
    
    
    return cb_model


# In[21]:


cb_params_views['iterations'] = 2500


# In[22]:


get_ipython().run_cell_magic('time', '', 'model_views_start = get_model(train_views_start, cb_params_views)')

2410        2410      0.514488     0.085848       0.996769      0.000571    57288.439777   21396.965895      4719.273203      150.040884
# In[23]:


#plot_feature_importance(cb_model_views, train_ds_views, 30)
plot_feature_importance(model_views_start.get_feature_importance(), train_views_start.get_feature_names(), 'CATBOOST')


# In[ ]:





# In[24]:


cb_params_views['iterations'] = 750


# In[25]:


get_ipython().run_cell_magic('time', '', 'model_views_end = get_model(train_views_end, cb_params_views)')

670         670      0.608364     0.065935       0.914413      0.008288     9189.254018     910.007358      4340.903897      127.150327
# In[26]:


plot_feature_importance(model_views_end.get_feature_importance(), train_views_end.get_feature_names(), 'CATBOOST')


# In[ ]:





# In[ ]:





# ## depth

# Обучаем модели для depth с разделением на модели start (данные до 2022-04-08) / end (данные после 2022-04-08) 

# In[27]:


cb_params_depth = cb_params_views


# In[28]:


cb_params_depth['iterations'] = 1600


# In[29]:


get_ipython().run_cell_magic('time', '', 'model_depth_start = get_model(train_depth_start, cb_params_depth)')

1550        1550      0.471369     0.152844        0.96694      0.002016         0.03689       0.003182         0.009499        0.000204
# In[30]:


#plot_feature_importance(cb_model_views, train_ds_views, 30)
plot_feature_importance(model_depth_start.get_feature_importance(), train_depth_start.get_feature_names(), 'CATBOOST')


# In[ ]:





# In[31]:


cb_params_depth['iterations'] = 2500


# In[32]:


get_ipython().run_cell_magic('time', '', 'model_depth_end = get_model(train_depth_end, cb_params_depth)')

2483        2483      0.339217     0.083196       0.981393      0.001967        0.017239        0.00378         0.002936        0.000174
# In[33]:


plot_feature_importance(model_depth_end.get_feature_importance(), train_depth_end.get_feature_names(), 'CATBOOST')


# In[ ]:





# In[ ]:





# In[ ]:





# ## full_reads_percent

# Обучаем модели для full_reads_percent с разделением на модели start (данные до 2022-04-08) / end (данные после 2022-04-08) 

# In[34]:


cb_params_frp = cb_params_views


# In[35]:


cb_params_frp['iterations'] = 1400


# In[36]:


get_ipython().run_cell_magic('time', '', 'model_frp_start = get_model(train_frp_start, cb_params_frp)')

1303        1303      0.556958     0.008903       0.948849      0.001106        7.007296       0.168132         2.382234         0.02588
# In[37]:


#plot_feature_importance(cb_model_views, train_ds_views, 30)
plot_feature_importance(model_frp_start.get_feature_importance(), train_frp_start.get_feature_names(), 'CATBOOST')


# In[ ]:





# In[38]:


cb_params_frp['iterations'] = 1000


# In[39]:


get_ipython().run_cell_magic('time', '', 'model_frp_end = get_model(train_frp_end, cb_params_frp)')

924         924      0.558665     0.027472       0.877142      0.002326        6.416195        0.07435         3.389853         0.04638
# In[40]:


#plot_feature_importance(cb_model_views, train_ds_views, 30)
plot_feature_importance(model_frp_end.get_feature_importance(), train_frp_end.get_feature_names(), 'CATBOOST')


# In[ ]:





# In[ ]:





# In[ ]:





# ## Сохраняем предсказания для ансамблей / стекинга

# In[ ]:





# In[78]:


pred_train = pd.DataFrame()
pred_train[['document_id', 'distrib_brdr']] = df_train[['document_id', 'distrib_brdr']]
pred_train = pred_train.reindex(['document_id', 'distrib_brdr', 'views', 'depth', 'full_reads_percent'], axis = 1)


# In[79]:


pred_train.loc[pred_train.query('distrib_brdr == 1').index, 'views'] = model_views_start.predict(df_train[df_train.distrib_brdr == 1][cat_cols + num_cols])
pred_train.loc[pred_train.query('distrib_brdr == 0').index, 'views'] = model_views_end.predict(df_train[df_train.distrib_brdr == 0][cat_cols + num_cols])
sum(pred_train.views.isna())


# In[80]:


pred_train.loc[pred_train.query('distrib_brdr == 1').index, 'depth'] = model_depth_start.predict(df_train[df_train.distrib_brdr == 1][cat_cols + num_cols])
pred_train.loc[pred_train.query('distrib_brdr == 0').index, 'depth'] = model_depth_end.predict(df_train[df_train.distrib_brdr == 0][cat_cols + num_cols])
sum(pred_train.depth.isna())


# In[81]:


pred_train.loc[pred_train.query('distrib_brdr == 1').index, 'full_reads_percent'] = model_frp_start.predict(df_train[df_train.distrib_brdr == 1][cat_cols + num_cols])
pred_train.loc[pred_train.query('distrib_brdr == 0').index, 'full_reads_percent'] = model_frp_end.predict(df_train[df_train.distrib_brdr == 0][cat_cols + num_cols])
sum(pred_train.full_reads_percent.isna())

pred_train.drop(['distrib_brdr'], axis =1, inplace = True)
pred_train.to_csv(os.path.join(DIR_SUBM_PART, f'{NAME}_train_part.csv'), index = False)
# In[ ]:





# ## Сохраняем модели

# In[ ]:





# In[82]:


#cb_model_views.save_model(os.path.join(DIR_MODELS, f'{NTRY}_pca64_cb_views.cbm'), 
model_views_start.save_model(os.path.join(DIR_MODELS, f'{NAME}_v_start.cbm'), 
                           format="cbm",
                           export_parameters=None,
                           pool=None
                         )
model_views_end.save_model(os.path.join(DIR_MODELS, f'{NAME}_v_end.cbm'), 
                           format="cbm",
                           export_parameters=None,
                           pool=None
                         )
model_depth_start.save_model(os.path.join(DIR_MODELS, f'{NAME}_d_start.cbm'), 
                           format="cbm",
                           export_parameters=None,
                           pool=None
                         )
model_depth_end.save_model(os.path.join(DIR_MODELS, f'{NAME}_d_end.cbm'), 
                           format="cbm",
                           export_parameters=None,
                           pool=None
                         )
model_frp_start.save_model(os.path.join(DIR_MODELS, f'{NAME}_f_start.cbm'), 
                           format="cbm",
                           export_parameters=None,
                           pool=None
                         )
model_frp_end.save_model(os.path.join(DIR_MODELS, f'{NAME}_f_end.cbm'), 
                           format="cbm",
                           export_parameters=None,
                           pool=None
                         )


# In[ ]:





# In[ ]:





# ## Делаем предсказание / сабмит

# In[83]:


subm = pd.DataFrame()
subm[['document_id', 'distrib_brdr']] = df_test[['document_id', 'distrib_brdr']]
subm = subm.reindex(['document_id', 'distrib_brdr', 'views', 'depth', 'full_reads_percent'], axis = 1)


# In[ ]:





# In[84]:


subm.loc[subm.query('distrib_brdr == 1').index, 'views'] = model_views_start.predict(df_test[df_test.distrib_brdr == 1][cat_cols + num_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'views'] = model_views_end.predict(df_test[df_test.distrib_brdr == 0][cat_cols + num_cols])
sum(subm.views.isna())


# In[85]:


subm.loc[subm.query('distrib_brdr == 1').index, 'depth'] = model_depth_start.predict(df_test[df_test.distrib_brdr == 1][cat_cols + num_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'depth'] = model_depth_end.predict(df_test[df_test.distrib_brdr == 0][cat_cols + num_cols])
sum(subm.depth.isna())


# In[86]:


subm.loc[subm.query('distrib_brdr == 1').index, 'full_reads_percent'] = model_frp_start.predict(df_test[df_test.distrib_brdr == 1][cat_cols + num_cols])
subm.loc[subm.query('distrib_brdr == 0').index, 'full_reads_percent'] = model_frp_end.predict(df_test[df_test.distrib_brdr == 0][cat_cols + num_cols])
sum(subm.full_reads_percent.isna())


# In[87]:


subm.drop(['distrib_brdr'], axis = 1, inplace = True)


# In[ ]:





# In[88]:


doc_id_ukr = df_test[df_test.spec == 1].document_id.values
subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[89]:


# присваиваем статичные данные
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'views'] = VIEWS_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'depth'] = DEPTH_UKR
subm.loc[subm.query('document_id in @doc_id_ukr').index, 'full_reads_percent'] = FPR_UKR

subm.query('document_id in @doc_id_ukr')[['views', 'depth', 'full_reads_percent']]


# In[ ]:





# In[90]:


subm.head()


# In[91]:


subm.to_csv(os.path.join(DIR_SUBM, f'{NAME}.csv'), index = False)


# In[ ]:





# In[92]:


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


# In[ ]:





# In[ ]:




