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

# но, например, для catboost не требуетмся такого кодирования, так что оригинальный признак так же останется в датасете,   
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





# In[6]:


df_train['publish_date'] = pd.to_datetime(df_train['publish_date'])
df_test['publish_date']  = pd.to_datetime(df_test['publish_date'])


# In[ ]:





# ## Очистка датасета

# этих категорий нет в тесте, а в трейне на них приходится всего 3 записи. они явно лишние.

# In[7]:


exclude_category = {'5e54e2089a7947f63a801742', '552e430f9a79475dd957f8b3', '5e54e22a9a7947f560081ea2' }


# In[8]:


df_train = df_train.query('category not in @exclude_category')
df_train.shape


# уберем статьи раньше минимальной даты в тесте. для начала так, дальше можно будет поиграться.

# In[9]:


#min_time = pd.Timestamp('2021-05-17')
min_time = df_test['publish_date'].min()


# In[10]:


df_train = df_train[df_train.publish_date > min_time]
df_train.shape


# In[ ]:





# In[ ]:





# In[ ]:





# ## title

# In[ ]:





# ## publish_date

# In[11]:


df_train['hour'] = df_train['publish_date'].dt.hour
df_train['dow']  = df_train['publish_date'].dt.dayofweek
#Monday=0, Sunday=6
df_train['weekend'] = (df_train.dow >= 4).astype(int) # 5
#df_train['holidays']
df_train['day']    = pd.to_datetime(df_train['publish_date']).dt.strftime("%d").astype(int)
df_train['mounth'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%m").astype(int)


# In[12]:


df_test['hour'] = df_test['publish_date'].dt.hour
df_test['dow']  = df_test['publish_date'].dt.dayofweek
#Monday=0, Sunday=6
df_test['weekend'] = (df_test.dow >= 4).astype(int) # 5
#df_train['holidays']
df_test['day']    = pd.to_datetime(df_test['publish_date']).dt.strftime("%d").astype(int)
df_test['mounth'] = pd.to_datetime(df_test['publish_date']).dt.strftime("%m").astype(int)


# In[ ]:





# In[13]:


df_train.drop('publish_date', axis = 1, inplace = True)
df_test.drop('publish_date', axis = 1, inplace = True)


# In[ ]:





# ## session

# In[ ]:





# ## authors

# авторы считываются как строки, а не как массив строк. исправим.

# In[14]:


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

# In[15]:


df_train['tags']  = df_train.tags.apply(lambda x: literal_eval(x))
df_test['tags']   = df_test.tags.apply( lambda x: literal_eval(x))


# In[ ]:





# In[ ]:





# разделяем категориальные и числовые признаки   
# числовые нормализуем

# In[16]:


df_train.columns


# In[17]:


num_cols = ['ctr']
cat_cols = ['hour', 'dow', 'weekend', 'day', 'mounth']


# ## normalize

# In[18]:


#scaler = preprocessing.MinMaxScaler()   #Transform features by scaling each feature to a given range.
#scaler = preprocessing.Normalizer()     #Normalize samples individually to unit norm.
scaler = preprocessing.StandardScaler()  #Standardize features by removing the mean and scaling to unit variance.

scaler.fit(df_train[num_cols])


# In[19]:


#df_train[num_cols].head(5)


# In[20]:


#df_test[num_cols].head(5)


# In[21]:


df_train[num_cols] = scaler.transform(df_train[num_cols])
df_test[num_cols]  = scaler.transform(df_test[num_cols])


# In[22]:


#df_train[num_cols].head(5)


# In[23]:


#df_test[num_cols].head(5)


# In[ ]:





# In[ ]:





# ## train_test_split

# вероятно лучше разделять до нормализации и категориальных энкодеров, что бы значения из валидационной выборки не были в учтены в тесте   
# однако, на первой итерации устроит и разбиение после всех преобразований

# In[24]:


x_train, x_val = train_test_split(df_train, test_size = 0.2)
df_train.shape, x_train.shape, x_val.shape


# In[ ]:





# ## save

# In[ ]:





# In[25]:


x_train.to_csv(os.path.join(DIR_DATA, 'x_train.csv'))
x_val.to_csv(os.path.join(DIR_DATA, 'x_val.csv'))
df_test.to_csv(os.path.join( DIR_DATA, 'test_upd.csv'))


# In[26]:


with open(os.path.join(DIR_DATA, 'num_columns.pkl'), 'wb') as pickle_file:
    pkl.dump(num_cols, pickle_file)


# In[27]:


with open(os.path.join(DIR_DATA, 'cat_columns.pkl'), 'wb') as pickle_file:
    pkl.dump(cat_cols, pickle_file)


# In[28]:


df_test.columns


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




