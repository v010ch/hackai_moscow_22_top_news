#!/usr/bin/env python
# coding: utf-8

# ## Загрузим нужные библиотеки

# In[67]:


import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


# Выполним загрузу датсета

# In[40]:


df_train = pd.read_csv("/content/train.csv", index_col= 0)
df_test = pd.read_csv("/content/test.csv", index_col= 0)


# ## Проанализируем датасет

# In[28]:


df_train.info()


# Заменим категорию и автора на число

# In[41]:


df_train["category"] = df_train["category"].astype('category')
df_train["category"] = df_train["category"].cat.codes
df_train["category"] = df_train["category"].astype('int')


# In[59]:


df_train["authors"] = df_train["authors"].astype('category')
df_train["authors"] = df_train["authors"].cat.codes
df_train["authors"] = df_train["authors"].astype('int')


# In[46]:


df_train['day'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%d").astype(int)
df_train['mounth'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%m").astype(int)


# In[47]:


df_train.head(3)


# In[49]:


features = list(set(df_train.columns) - set(['publish_date']))

_ = df_train[features].hist(figsize=(20,12))


# Всего 9 категорий статей

# In[51]:


df_train.category.value_counts()


# ## Выделим выборки

# In[62]:


X = df_train.drop(["views","depth","full_reads_percent","title","publish_date", "session", "tags"], axis = 1)
y = df_train[["views","depth","full_reads_percent"]]


# In[64]:


X.head()


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## Подбор модели

# In[68]:


regr = RandomForestRegressor(random_state=0)


# Обучим модель

# In[69]:


regr.fit(X_train, y_train)


# Предскажем значения

# In[72]:


pred = regr.predict(X_test)


# ## Оценка точности

# In[82]:


score_views = r2_score(y_test["views"], pred[:,0])
score_depth = r2_score(y_test["depth"], pred[:,1])
score_frp = r2_score(y_test["full_reads_percent"], pred[:,2])


# In[83]:


score = 0.4 * score_views + 0.3 * score_depth + 0.3 * score_frp

score

