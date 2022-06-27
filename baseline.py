#!/usr/bin/env python
# coding: utf-8

# ## Загрузим нужные библиотеки

# In[3]:


import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


# ### Reproducibility block

# In[4]:


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


# Выполним загрузу датсета

# In[8]:


DIR_DATA  = os.path.join(os.getcwd(), 'data')
#DIR_TRAIN = os.path.join(DIR_DATA, 'train')
#DIR_TEST  = os.path.join(DIR_DATA, 'test')
DIR_SUBM  = os.path.join(os.getcwd(), 'subm')


# In[ ]:





# In[11]:


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'), index_col= 0)
df_test = pd.read_csv(os.path.join(DIR_DATA, 'test.csv'), index_col= 0)


# ## Проанализируем датасет

# In[12]:


df_train.info()


# Заменим категорию и автора на число

# In[13]:


df_train["category"] = df_train["category"].astype('category')
df_train["category"] = df_train["category"].cat.codes
df_train["category"] = df_train["category"].astype('int')


# In[14]:


df_train["authors"] = df_train["authors"].astype('category')
df_train["authors"] = df_train["authors"].cat.codes
df_train["authors"] = df_train["authors"].astype('int')


# In[15]:


df_train['day'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%d").astype(int)
df_train['mounth'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%m").astype(int)


# In[17]:


df_train.head(3)


# In[18]:


features = list(set(df_train.columns) - set(['publish_date']))

_ = df_train[features].hist(figsize=(20,12))


# Всего 9 категорий статей

# In[19]:


df_train.category.value_counts()


# ## Выделим выборки

# In[20]:


X = df_train.drop(["views","depth","full_reads_percent","title","publish_date", "session", "tags"], axis = 1)
y = df_train[["views","depth","full_reads_percent"]]


# In[21]:


X.head()


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## Подбор модели

# In[23]:


regr = RandomForestRegressor(random_state=0)


# Обучим модель

# In[24]:


regr.fit(X_train, y_train)


# Предскажем значения

# In[25]:


pred = regr.predict(X_test)


# ## Оценка точности

# In[26]:


score_views = r2_score(y_test["views"], pred[:,0])
score_depth = r2_score(y_test["depth"], pred[:,1])
score_frp = r2_score(y_test["full_reads_percent"], pred[:,2])


# In[27]:


score = 0.4 * score_views + 0.3 * score_depth + 0.3 * score_frp

score


# In[ ]:





# In[ ]:





# Предсказание для теста

# In[29]:


df_test["category"] = df_test["category"].astype('category')
df_test["category"] = df_test["category"].cat.codes
df_test["category"] = df_test["category"].astype('int')


# In[30]:


df_test["authors"] = df_test["authors"].astype('category')
df_test["authors"] = df_test["authors"].cat.codes
df_test["authors"] = df_test["authors"].astype('int')


# In[31]:


df_test['day']    = pd.to_datetime(df_test['publish_date']).dt.strftime("%d").astype(int)
df_test['mounth'] = pd.to_datetime(df_test['publish_date']).dt.strftime("%m").astype(int)


# In[ ]:





# In[36]:


X_pred = df_test.drop(["title","publish_date", "session", "tags"], axis = 1)


# In[40]:


X_pred.head(3)


# In[ ]:





# In[41]:


pred = regr.predict(X_pred)


# In[ ]:





# submission

# In[45]:


subm = pd.read_csv(os.path.join(DIR_SUBM, 'sample_solution.csv'))
subm.shape


# In[46]:


subm.head()


# In[53]:


subm.document_id = df_test.index
subm.views = pred[:,0]
subm.depth = pred[:,1]
subm.full_reads_percent = pred[:,2]


#y_test["views"], pred[:,0]
#y_test["depth"], pred[:,1]
#y_test["full_reads_percent"], pred[:,2]


# In[54]:


subm.to_csv(os.path.join(DIR_SUBM, '0_baseline.csv'), index = False)


# In[ ]:





# In[ ]:




