#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle as pkl

import pandas as pd
from sklearn.cluster import KMeans#, SpectralClustering

import seaborn as sns


# In[ ]:


sns.set(rc={'figure.figsize':(30,16)}) # Setting seaborn as default style even if use only matplotlib
sns.set(font_scale = 2)


# In[ ]:





# In[ ]:


DIR_DATA  = os.path.join(os.getcwd(), 'data')


# In[ ]:





# In[ ]:





# In[ ]:


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'))
df_train.shape


# depth явно состоих из 2х распределений 

# In[ ]:


df_train.depth.hist(bins = 40, figsize = (24, 12) )


# разобьем на 2 класса (+ выбросы?) по имеющимся у нас значениям depth.   
# далее, итеративно, по различныи признакам будем классифицировать на основе вышерасчитанного разбиения.   
# после чего depth будем считать для этих 2х классов в отдельности по отдельным алгоритмам

# In[ ]:





# In[ ]:





# In[ ]:


kmeans2 = KMeans(n_clusters=2, random_state=0).fit(df_train.depth.values.reshape(-1, 1))
df_train['c2'] = kmeans2.predict(df_train.depth.values.reshape(-1, 1))
df_train.c2.value_counts()


# In[ ]:


sns.histplot(data = df_train[['depth', 'c2']], x = 'depth', hue="c2", )


# In[ ]:





# In[ ]:





# In[ ]:


kmeans3 = KMeans(n_clusters=3, random_state=0).fit(df_train.depth.values.reshape(-1, 1))
df_train['c3'] = kmeans3.predict(df_train.depth.values.reshape(-1, 1))
df_train.c3.value_counts()


# In[ ]:


sns.histplot(data = df_train[['depth', 'c3']], x = 'depth', hue="c3", )


# In[ ]:





# In[ ]:





# Сохраним классы для анализа

# In[ ]:


df_train.to_csv(os.path.join(DIR_DATA, 'train_depth_classes.csv'), index = False)


# In[ ]:




