#!/usr/bin/env python
# coding: utf-8

# ## Загрузим нужные библиотеки

# In[1]:


import os
import numpy as np
import pandas as pd

from ast import literal_eval


# In[2]:


#import plotly.express as px

import seaborn as sns
from matplotlib import pyplot as plt
sns.set() # Setting seaborn as default style even if use only matplotlib


# ### Reproducibility block

# In[3]:


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





# In[ ]:





# In[ ]:


def plot_hists_sns(inp_df, inp_feature):
    
    fig, axes = plt.subplots(4, 3, figsize=(30,30))
    
    sns.histplot(ax = axes[0, 0],
                data = inp_df,
                x = inp_feature
                )
    
    
#views
    tmp_df = inp_df.groupby([inp_feature]).views.agg(val_min='min', val_max='max', val_aver='mean')
    sns.barplot(ax = axes[1, 0],
               x = tmp_df.index,
               y = tmp_df.val_min.values
               )
    axes[1, 0].set_title(f'Histogram of minimum views over {inp_feature}')
    sns.barplot(ax = axes[1, 1],
               x = tmp_df.index,
               y = tmp_df.val_max.values
               )
    axes[1, 1].set_title(f'Histogram of maximum views over {inp_feature}')
    sns.barplot(ax = axes[1, 2],
               x = tmp_df.index,
               y = tmp_df.val_aver.values
               )
    axes[1, 2].set_title(f'Histogram of average views over {inp_feature}')
    
    
#depth
    tmp_df = inp_df.groupby([inp_feature]).depth.agg(val_min='min', val_max='max', val_aver='mean')
    sns.barplot(ax = axes[2, 0],
               x = tmp_df.index,
               y = tmp_df.val_min.values
               )
    axes[2, 0].set_title(f'Histogram of minimum depth over {inp_feature}')
    sns.barplot(ax = axes[2, 1],
               x = tmp_df.index,
               y = tmp_df.val_max.values
               )
    axes[2, 1].set_title(f'Histogram of maximum depth over {inp_feature}')
    sns.barplot(ax = axes[2, 2],
               x = tmp_df.index,
               y = tmp_df.val_aver.values
               )
    axes[2, 2].set_title(f'Histogram of average depth over {inp_feature}')
    
    
#full_reads_percent
    tmp_df = inp_df.groupby([inp_feature]).full_reads_percent.agg(val_min='min', val_max='max', val_aver='mean')
    sns.barplot(ax = axes[3, 0],
               x = tmp_df.index,
               y = tmp_df.val_min.values
               )
    axes[3, 0].set_title(f'Histogram of minimum full_reads_percent over {inp_feature}')
    sns.barplot(ax = axes[3, 1],
               x = tmp_df.index,
               y = tmp_df.val_max.values
               )
    axes[3, 1].set_title(f'Histogram of maximum full_reads_percent over {inp_feature}')
    sns.barplot(ax = axes[3, 2],
               x = tmp_df.index,
               y = tmp_df.val_aver.values
               )
    axes[3, 2].set_title(f'Histogram of aver full_reads_percent over {inp_feature}')
    
    fig.show()


# In[ ]:





# In[ ]:





# Выполним загрузу датсета

# In[4]:


DIR_DATA  = os.path.join(os.getcwd(), 'data')
#DIR_TRAIN = os.path.join(DIR_DATA, 'train')
#DIR_TEST  = os.path.join(DIR_DATA, 'test')
DIR_SUBM  = os.path.join(os.getcwd(), 'subm')


# In[ ]:





# In[5]:


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'), index_col= 0)
df_test = pd.read_csv(os.path.join(DIR_DATA, 'test.csv'), index_col= 0)


# In[ ]:





# In[ ]:





# In[ ]:





# # Проанализируем датасет

# In[6]:


df_train.info()


# In[7]:


df_train.describe()


# ● **document id** - идентификатор    
# ● **title** - заголовок статьи   
# ● **publish_date** - время публикации   
# ● **session** - номер сессии   
# ● **authors** - код автора   
# ● **views** - количество просмотров   
# ● **depth** - объем прочитанного материала   
# ● **full_reads percent** - процент читателей полностью прочитавших статью   
# ● **ctr** - показатель кликабельности   
# ● **category** - категория статьи   
# ● **tags** - ключевые слова в статье   

# In[8]:


df_train.shape, df_train.index.nunique()


# In[ ]:





# # publish_date

# In[9]:


df_train['publish_date'] = pd.to_datetime(df_train['publish_date'])

df_train['hour'] = df_train['publish_date'].dt.hour
df_train['dow'] = df_train['publish_date'].dt.dayofweek
#Monday=0, Sunday=6
df_train['weekend'] = (df_train.dow >= 4) # 5
#df_train['holidays']
df_train['day'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%d").astype(int)
df_train['mounth'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%m").astype(int)


# In[115]:


df_test['publish_date'] = pd.to_datetime(df_test['publish_date'])


# проверим границы дат

# In[117]:


df_train['publish_date'].min(), df_test['publish_date'].min()


# In[118]:


df_train['publish_date'].max(), df_test['publish_date'].max()


# In[120]:


df_train[df_train.publish_date > df_test['publish_date'].min()].shape


# In[121]:


df_train[df_train.publish_date < df_test['publish_date'].min()].shape


# всего 6 статей в трейне датой раньше, чем минимальная дата в тесте. вероятно их следует исключить исходя из предположения, что они из другого распределения

# In[125]:


#df_train[df_train.publish_date < df_test['publish_date'].min()]


# In[ ]:





# In[88]:


plot_hists_sns(df_train, 'dow')


# In[89]:


plot_hists_sns(df_train, 'hour')


# In[90]:


plot_hists_sns(df_train, 'day')


# In[91]:


plot_hists_sns(df_train, 'mounth')


# In[129]:


plot_hists_sns(df_train, 'weekend')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # category

# In[99]:


df_train.category.nunique(), df_train.category.unique(), 


# In[100]:


plot_hists_sns(df_train, 'category')


# In[101]:


df_train.category.value_counts()


# In[110]:


df_test.category.value_counts()


# вероятно стоит удалить последние 3 категории, что бы модель не переобучалась на них. к тому же их нет в тесте

# In[106]:


exclude_category = {'5e54e2089a7947f63a801742', '552e430f9a79475dd957f8b3', '5e54e22a9a7947f560081ea2' }


# In[104]:


plot_hists_sns(df_train.query('category in @exclude_category'), 'category')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## authors

# In[176]:


df_train['authors']  = df_train.authors.apply(lambda x: literal_eval(x))
df_train['Nauthors'] = df_train.authors.apply(lambda x: len(x))

df_test['authors']  = df_test.authors.apply(lambda x: literal_eval(x))
df_test['Nauthors'] = df_test.authors.apply(lambda x: len(x))


# In[177]:


df_train['Nauthors'].value_counts()


# In[178]:


df_test['Nauthors'].value_counts()


# удивительно, что возможные значения количества авторов в трейне и тесте совпадают. можно использовать как признак  
# однако значения при > 3 малы, что может привести к переобучению

# In[175]:


plot_hists_sns(df_train, 'Nauthors')


# In[194]:


df_train['Nauthors_upd'] = df_train['Nauthors'].apply(lambda x: x if x < 4 else 4) # 3


# In[195]:


df_train['Nauthors_upd'].value_counts()


# In[196]:


plot_hists_sns(df_train, 'Nauthors_upd')


# In[197]:


all_authors = set()
for el in df_train.authors.values:
    if len (el) == 0:
        continue
    if len(el) == 1:
        all_authors.add(el[0])
        continue
        
    for author in el:
        all_authors.add(author)


# In[182]:


len(all_authors)


# In[208]:


all_authors_test = set()
for el in df_test.authors.values:
    if len (el) == 0:
        continue
    if len(el) == 1:
        all_authors_test.add(el[0])
        continue
        
    for author in el:
        all_authors_test.add(author)


# In[209]:


len(all_authors_test)


# In[213]:


missed_authors = set()
for el in all_authors_test:
    if el not in all_authors:
        missed_authors.add(el)


# In[215]:


len(missed_authors)


# только 2 (2%) автора не представленны в обучающей выборке

# In[ ]:





# In[ ]:





# In[ ]:





# ## tags

# In[201]:


df_train['tags']  = df_train.tags.apply(lambda x: literal_eval(x))
df_train['Ntags'] = df_train.tags.apply(lambda x: len(x))

df_test['tags']  = df_test.tags.apply(lambda x: literal_eval(x))
df_test['Ntags'] = df_test.tags.apply(lambda x: len(x))


# In[202]:


df_train.Ntags.value_counts()


# In[203]:


df_test.Ntags.value_counts()


# в тест есть статьи с большим количеством тэгов чем в трейне. хоть их количество и мало

# In[210]:


plot_hists_sns(df_train, 'Ntags')


# In[ ]:





# In[204]:


all_tags = set()
for el in df_train.tags.values:
    if len (el) == 0:
        continue
    if len(el) == 1:
        all_tags.add(el[0])
        continue
        
    for tag in el:
        all_tags.add(tag)


# In[205]:


len(all_tags)


# In[206]:


all_tags_test = set()
for el in df_test.tags.values:
    if len (el) == 0:
        continue
    if len(el) == 1:
        all_tags_test.add(el[0])
        continue
        
    for tag in el:
        all_tags_test.add(tag)


# In[207]:


len(all_tags_test)


# In[216]:


missed_tags = set()
for el in all_tags_test:
    if el not in all_tags:
        missed_tags.add(el)


# In[217]:


len(missed_tags)


# 1149 (17%) тэгов не представлены в обучающей выборке!!!!!

# In[ ]:





# In[ ]:





# ## ctr

# In[221]:


df_train.hist('ctr')


# In[222]:


df_train.ctr.min(), df_train.ctr.max()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


df_train.columns


# In[15]:



#df_train['day'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%d").astype(int)
#df_train['mounth'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%m").astype(int)


# In[23]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:





# In[29]:


df_train.authors


# In[30]:


df_train.shape


# In[33]:


df_train.index.nunique()


# In[ ]:




