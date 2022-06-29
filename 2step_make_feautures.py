#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle as pkl

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from ast import literal_eval


# ## Reproducibility block

# In[2]:


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


# In[ ]:





# In[3]:


DIR_DATA  = os.path.join(os.getcwd(), 'data')


# энкодеры для кодирования категориальных переменных. 

# но, например, для catboost не требуется такого кодирования, так что оригинальный признак так же останется в датасете,   
# а в модель будут передоваться признаки только через параметр features.
all_encoders = [ce.BackwardDifferenceEncoder(), 
ce.BaseNEncoder(), 
ce.BinaryEncoder(),
ce.CatBoostEncoder(),
ce.CountEncoder(),
ce.GLMMEncoder(),
ce.HashingEncoder(),
ce.HelmertEncoder(),
ce.JamesSteinEncoder(),
ce.LeaveOneOutEncoder(),
ce.MEstimateEncoder(),
ce.OneHotEncoder(),
ce.OrdinalEncoder(),
ce.SumEncoder(),
ce.PolynomialEncoder(),
ce.TargetEncoder(),
ce.WOEEncoder(),
#ce.QuantileEncoder(),
]
# In[ ]:





# In[4]:


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'), index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test.csv'), index_col= 0)


# In[5]:


df_train.shape, df_test.shape


# In[ ]:





# Добавляем эмбединги

# In[6]:


# should try and without it
clean_text = lambda x:' '.join(re.sub('\n|\r|\t|[^а-я]', ' ', x.lower()).split())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


df_train['publish_date'] = pd.to_datetime(df_train['publish_date'])
df_test['publish_date']  = pd.to_datetime(df_test['publish_date'])


# In[ ]:





# ## Очистка датасета

# этих категорий нет в тесте, а в трейне на них приходится всего 3 записи. они явно лишние.

# In[8]:


exclude_category = {'5e54e2089a7947f63a801742', '552e430f9a79475dd957f8b3', '5e54e22a9a7947f560081ea2' }


# In[9]:


df_train = df_train.query('category not in @exclude_category')
df_train.shape


# уберем статьи раньше минимальной даты в тесте. для начала так, дальше можно будет поиграться.

# In[10]:


#min_time = pd.Timestamp('2021-05-17')
min_time = df_test['publish_date'].min()


# In[11]:


df_train = df_train[df_train.publish_date > min_time]
df_train.shape


# In[ ]:





# In[ ]:





# In[ ]:





# ## title

# In[ ]:





# ## publish_date

# In[12]:


df_train['hour'] = df_train['publish_date'].dt.hour
df_train['dow']  = df_train['publish_date'].dt.dayofweek
#Monday=0, Sunday=6
df_train['weekend'] = (df_train.dow >= 4).astype(int) # 5
#df_train['holidays']
df_train['day']    = pd.to_datetime(df_train['publish_date']).dt.strftime("%d").astype(int)
df_train['mounth'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%m").astype(int)


# In[13]:


df_test['hour'] = df_test['publish_date'].dt.hour
df_test['dow']  = df_test['publish_date'].dt.dayofweek
#Monday=0, Sunday=6
df_test['weekend'] = (df_test.dow >= 4).astype(int) # 5
#df_train['holidays']
df_test['day']    = pd.to_datetime(df_test['publish_date']).dt.strftime("%d").astype(int)
df_test['mounth'] = pd.to_datetime(df_test['publish_date']).dt.strftime("%m").astype(int)


# In[ ]:





# In[14]:


df_train.drop('publish_date', axis = 1, inplace = True)
df_test.drop('publish_date', axis = 1, inplace = True)


# In[ ]:





# ## session

# In[ ]:





# ## authors

# авторы считываются как строки, а не как массив строк. исправим.

# In[15]:


df_train['authors']  = df_train.authors.apply(lambda x: literal_eval(x))
df_test['authors']   = df_test.authors.apply( lambda x: literal_eval(x))


# In[ ]:





# In[ ]:





# In[ ]:





# ## ctr

# In[ ]:





# ## category

# In[ ]:





# ## tags

# In[16]:


df_train['tags']  = df_train.tags.apply(lambda x: literal_eval(x))
df_test['tags']   = df_test.tags.apply( lambda x: literal_eval(x))


# In[ ]:





# In[ ]:





# разделяем категориальные и числовые признаки   
# числовые нормализуем

# In[17]:


df_train.columns


# In[18]:


num_cols = ['ctr']
cat_cols = ['hour', 'dow', 'weekend', 'day', 'mounth']


# ## normalize

# In[19]:


#scaler = preprocessing.MinMaxScaler()   #Transform features by scaling each feature to a given range.
#scaler = preprocessing.Normalizer()     #Normalize samples individually to unit norm.
scaler = preprocessing.StandardScaler()  #Standardize features by removing the mean and scaling to unit variance.

scaler.fit(df_train[num_cols])


# In[20]:


#df_train[num_cols].head(5)


# In[21]:


#df_test[num_cols].head(5)


# In[22]:


df_train[num_cols] = scaler.transform(df_train[num_cols])
df_test[num_cols]  = scaler.transform(df_test[num_cols])


# In[23]:


#df_train[num_cols].head(5)


# In[24]:


#df_test[num_cols].head(5)


# In[ ]:





# Добавляем эмбединги

# In[25]:


# sberbank-ai/sbert_large_mt_nlu_ru       1024  1.71Gb
# DeepPavlov/rubert-base-cased-sentence   768   0.7Gb
# DeepPavlov/rubert-base-cased-conversational  768
# DeepPavlov/rubert-base-cased            768
# sberbank-ai/sbert_large_nlu_ru          1024  1.71Gb

MODEL_FOLDER = 'sbert_large_mt_nlu_ru'
MAX_LENGTH = 24


# In[26]:


emb_train = pd.read_csv(os.path.join(DIR_DATA, f'ttl_emb_train_{MODEL_FOLDER}_{MAX_LENGTH}.csv'))
#emb_train.drop(['document_id', 'title'], axis = 1 , inplace = True)
emb_train.drop(['title'], axis = 1 , inplace = True)

df_train = df_train.merge(emb_train, on = 'document_id', validate = 'one_to_one')
df_train.shape, emb_train.shape


# In[27]:


emb_test = pd.read_csv(os.path.join(DIR_DATA, f'ttl_emb_test_{MODEL_FOLDER}_{MAX_LENGTH}.csv'))
#emb_test.drop(['document_id', 'title'], axis = 1 , inplace = True)
emb_test.drop(['title'], axis = 1 , inplace = True)

df_test = df_test.merge(emb_test, on = 'document_id', validate = 'one_to_one')
df_test.shape, emb_test.shape


# In[28]:


num_cols = ['ctr'] + list(emb_train.columns)


# In[35]:


if 'document_id' in num_cols:
    num_cols.remove('document_id')


# In[ ]:





# ## train_test_split

# вероятно лучше разделять до нормализации и категориальных энкодеров, что бы значения из валидационной выборки не были в учтены в тесте   
# однако, на первой итерации устроит и разбиение после всех преобразований

# In[37]:


x_train, x_val = train_test_split(df_train, test_size = 0.2)
df_train.shape, x_train.shape, x_val.shape


# In[ ]:





# ## save

# In[ ]:





# In[38]:


x_train.to_csv(os.path.join(DIR_DATA,  'x_train.csv'))
x_val.to_csv(os.path.join(DIR_DATA,    'x_val.csv'))
df_test.to_csv(os.path.join( DIR_DATA, 'test_upd.csv'))


# In[39]:


with open(os.path.join(DIR_DATA, 'num_columns.pkl'), 'wb') as pickle_file:
    pkl.dump(num_cols, pickle_file)


# In[40]:


with open(os.path.join(DIR_DATA, 'cat_columns.pkl'), 'wb') as pickle_file:
    pkl.dump(cat_cols, pickle_file)


# In[41]:


df_test.columns


# In[ ]:





# In[ ]:




