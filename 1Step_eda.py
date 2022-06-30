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
sns.set(rc={'figure.figsize':(30,16)}) # Setting seaborn as default style even if use only matplotlib
sns.set(font_scale = 2)
plt.rcParams['figure.figsize']=(30,16)


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





# In[4]:


def plot_hists_sns(inp_df, inp_feature):
    
    fig, axes = plt.subplots(4, 3, figsize=(30,30))
    
    sns.histplot(ax = axes[0, 0],
                data = inp_df,
                x = inp_feature,
                #hue = 'c2'
                )
    
    
#views
    tmp_df = inp_df.groupby([inp_feature]).views.agg(val_min='min', val_max='max', val_aver='mean')
    sns.barplot(ax = axes[1, 0],
               x = tmp_df.index,
               y = tmp_df.val_min.values,
                #hue = 'c2'
               )
    axes[1, 0].set_title(f'Histogram of minimum views over {inp_feature}')
    sns.barplot(ax = axes[1, 1],
               x = tmp_df.index,
               y = tmp_df.val_max.values,
                #hue = 'c2'
               )
    axes[1, 1].set_title(f'Histogram of maximum views over {inp_feature}')
    sns.barplot(ax = axes[1, 2],
               x = tmp_df.index,
               y = tmp_df.val_aver.values,
                #hue = 'c2'
               )
    axes[1, 2].set_title(f'Histogram of average views over {inp_feature}')
    
    
#depth
    tmp_df = inp_df.groupby([inp_feature]).depth.agg(val_min='min', val_max='max', val_aver='mean')
    sns.barplot(ax = axes[2, 0],
               x = tmp_df.index,
               y = tmp_df.val_min.values,
                #hue = 'c2'
               )
    axes[2, 0].set_title(f'Histogram of minimum depth over {inp_feature}')
    sns.barplot(ax = axes[2, 1],
               x = tmp_df.index,
               y = tmp_df.val_max.values,
                #hue = 'c2'
               )
    axes[2, 1].set_title(f'Histogram of maximum depth over {inp_feature}')
    sns.barplot(ax = axes[2, 2],
               x = tmp_df.index,
               y = tmp_df.val_aver.values,
                #hue = 'c2'
               )
    axes[2, 2].set_title(f'Histogram of average depth over {inp_feature}')
    
    
#full_reads_percent
    tmp_df = inp_df.groupby([inp_feature]).full_reads_percent.agg(val_min='min', val_max='max', val_aver='mean')
    sns.barplot(ax = axes[3, 0],
               x = tmp_df.index,
               y = tmp_df.val_min.values,
                #hue = 'c2'
               )
    axes[3, 0].set_title(f'Histogram of minimum full_reads_percent over {inp_feature}')
    sns.barplot(ax = axes[3, 1],
               x = tmp_df.index,
               y = tmp_df.val_max.values,
                #hue = 'c2'
               )
    axes[3, 1].set_title(f'Histogram of maximum full_reads_percent over {inp_feature}')
    sns.barplot(ax = axes[3, 2],
               x = tmp_df.index,
               y = tmp_df.val_aver.values,
                #hue = 'c2'
               )
    axes[3, 2].set_title(f'Histogram of aver full_reads_percent over {inp_feature}')
    
    fig.show()


# In[5]:


def plot_corrc(inp_df, inp_cols, targ_cols = ['views', 'depth', 'full_reads_percent']):
    f, ax = plt.subplots(1, 2, figsize=(24, 8))
    sns.heatmap(df_train[inp_cols + targ_cols].corr(), 
    #sns.heatmap(df_train.query('c2 == 0')[inp_cols + targ_cols].corr(), 
                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[0])
    sns.heatmap(df_train[inp_cols + targ_cols].corr(), 
    #sns.heatmap(df_train.query('c2 == 1')[inp_cols + targ_cols].corr(), 
                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[1])
#    sns.heatmap(df_train.query('c2 == 0')[inp_cols + targ_cols].corr(method = 'spearman'), 
#                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[1, 0])
#    sns.heatmap(df_train.query('c2 == 1')[inp_cols + targ_cols].corr(method = 'spearman'), 
#                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[1, 1])
    
    sns.pairplot(df_train[inp_cols + targ_cols], height = 16,) #hue = 'c2')


# In[ ]:





# Выполним загрузу датсета

# In[6]:


DIR_DATA  = os.path.join(os.getcwd(), 'data')
#DIR_TRAIN = os.path.join(DIR_DATA, 'train')
#DIR_TEST  = os.path.join(DIR_DATA, 'test')
DIR_SUBM  = os.path.join(os.getcwd(), 'subm')


# In[ ]:





# In[7]:


#df_train = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'), index_col= 0)
df_train = pd.read_csv(os.path.join(DIR_DATA, 'train_depth_classes.csv'), index_col= 0)

df_test = pd.read_csv(os.path.join(DIR_DATA, 'test.csv'), index_col= 0)


# In[ ]:





# In[ ]:





# In[ ]:





# # Проанализируем датасет

# In[8]:


df_train.info()


# In[9]:


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

# In[10]:


df_train.shape, df_train.index.nunique()


# In[11]:


df_train.columns


# ## Targets

# In[ ]:





# In[12]:


plot_corrc(df_train, ['views'], ['depth', 'full_reads_percent'])


# Ожидаемо views может быть применено как фича для других таргетов.   
# depth из состоит 2х распределений. необходимы 2 модели.    
# между frp и двумя головами depth наглядно имеется корреляция.

# In[ ]:





# # publish_date

# In[13]:


df_train['publish_date'] = pd.to_datetime(df_train['publish_date'])

df_train['hour'] = df_train['publish_date'].dt.hour
df_train['dow'] = df_train['publish_date'].dt.dayofweek
#Monday=0, Sunday=6
df_train['weekend'] = (df_train.dow >= 4) # 5
#df_train['holidays']
df_train['day'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%d").astype(int)
df_train['mounth'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%m").astype(int)


# In[14]:


df_test['publish_date'] = pd.to_datetime(df_test['publish_date'])


# проверим границы дат

# In[15]:


df_train['publish_date'].min(), df_test['publish_date'].min()


# In[16]:


df_train['publish_date'].max(), df_test['publish_date'].max()


# In[17]:


df_train[df_train.publish_date > df_test['publish_date'].min()].shape


# In[18]:


df_train[df_train.publish_date < df_test['publish_date'].min()].shape


# In[19]:


#df_train.sort_values(by='publish_date').head(15)


# In[20]:


#df_test.sort_values(by='publish_date').head(15)


# в тесте скачек с 2021-10-04 на 2022-01-31 21:00:26. Далее даты идут регулярно.

# всего 6 статей в трейне датой раньше, чем минимальная дата в тесте.   
# далее скачек: 2021-05-17, 2021-09-17, 2021-12-17, 2021-12-17, 2022-01-29 06:00:22.   
# в целом все даты 21 года выглядят оторванными, однако, могут находится в том же распределении, так что, вероятно, их стоит оставить.   
# статьи из 17 и 18 гг, вероятно стоит исключить после проверки на наличия в них особенных авторов или тэгов.

# In[21]:


#df_train[df_train.publish_date < df_test['publish_date'].min()]


# In[ ]:





# In[22]:


plot_hists_sns(df_train, 'dow')


# In[23]:


plot_hists_sns(df_train, 'hour')


# In[24]:


plot_hists_sns(df_train, 'day')


# In[25]:


plot_hists_sns(df_train, 'mounth')


# In[26]:


plot_hists_sns(df_train, 'weekend')


# In[ ]:





# In[70]:


plot_corrc(df_train, ['hour', 'dow', 'day', 'mounth'],['views', 'depth', 'full_reads_percent']) #'weekend', 


# In[65]:


df_train.columns


# In[ ]:





# In[ ]:





# # category

# In[27]:


df_train.category.nunique(), df_train.category.unique(), 


# In[28]:


plot_hists_sns(df_train, 'category')


# In[29]:


df_train.category.value_counts()


# In[30]:


df_test.category.value_counts()


# вероятно стоит удалить последние 3 категории, что бы модель не переобучалась на них. к тому же их нет в тесте

# In[31]:


exclude_category = {'5e54e2089a7947f63a801742', '552e430f9a79475dd957f8b3', '5e54e22a9a7947f560081ea2' }


# In[32]:


plot_hists_sns(df_train.query('category in @exclude_category'), 'category')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## authors

# In[33]:


df_train['authors']  = df_train.authors.apply(lambda x: literal_eval(x))
df_train['Nauthors'] = df_train.authors.apply(lambda x: len(x))

df_test['authors']  = df_test.authors.apply(lambda x: literal_eval(x))
df_test['Nauthors'] = df_test.authors.apply(lambda x: len(x))


# In[34]:


df_train['Nauthors'].value_counts()


# In[35]:


df_test['Nauthors'].value_counts()


# удивительно, что возможные значения количества авторов в трейне и тесте совпадают. можно использовать как признак  
# однако значения при > 3 малы, что может привести к переобучению

# In[36]:


plot_hists_sns(df_train, 'Nauthors')


# In[37]:


df_train['Nauthors_upd'] = df_train['Nauthors'].apply(lambda x: x if x < 4 else 4) # 3


# In[38]:


df_train['Nauthors_upd'].value_counts()


# In[39]:


plot_hists_sns(df_train, 'Nauthors_upd')


# In[40]:


all_authors = set()
for el in df_train.authors.values:
    if len (el) == 0:
        continue
    if len(el) == 1:
        all_authors.add(el[0])
        continue
        
    for author in el:
        all_authors.add(author)


# In[41]:


len(all_authors)


# In[42]:


all_authors_test = set()
for el in df_test.authors.values:
    if len (el) == 0:
        continue
    if len(el) == 1:
        all_authors_test.add(el[0])
        continue
        
    for author in el:
        all_authors_test.add(author)


# In[43]:


len(all_authors_test)


# In[44]:


missed_authors = set()
for el in all_authors_test:
    if el not in all_authors:
        missed_authors.add(el)


# In[45]:


len(missed_authors)


# только 2 (2%) автора не представленны в обучающей выборке

# In[67]:


plot_corrc(df_train, ['Nauthors'], ['views', 'depth', 'full_reads_percent'])


# In[ ]:





# In[ ]:





# ## tags

# In[46]:


df_train['tags']  = df_train.tags.apply(lambda x: literal_eval(x))
df_train['Ntags'] = df_train.tags.apply(lambda x: len(x))

df_test['tags']  = df_test.tags.apply(lambda x: literal_eval(x))
df_test['Ntags'] = df_test.tags.apply(lambda x: len(x))


# In[47]:


df_train.Ntags.value_counts()


# In[48]:


df_test.Ntags.value_counts()


# в тест есть статьи с большим количеством тэгов чем в трейне. хоть их количество и мало

# In[49]:


plot_hists_sns(df_train, 'Ntags')


# In[ ]:





# In[50]:


all_tags = set()
for el in df_train.tags.values:
    if len (el) == 0:
        continue
    if len(el) == 1:
        all_tags.add(el[0])
        continue
        
    for tag in el:
        all_tags.add(tag)


# In[51]:


len(all_tags)


# In[52]:


all_tags_test = set()
for el in df_test.tags.values:
    if len (el) == 0:
        continue
    if len(el) == 1:
        all_tags_test.add(el[0])
        continue
        
    for tag in el:
        all_tags_test.add(tag)


# In[53]:


len(all_tags_test)


# In[54]:


missed_tags = set()
for el in all_tags_test:
    if el not in all_tags:
        missed_tags.add(el)


# In[55]:


len(missed_tags)


# 1149 (17%) тэгов не представлены в обучающей выборке!!!!!

# In[ ]:





# In[ ]:





# ## ctr

# In[56]:


df_train.hist('ctr', bins = 40, figsize=(24, 8))


# In[57]:


df_train.ctr.min(), df_train.ctr.max()


# In[66]:


plot_corrc(df_train, ['ctr'], ['views', 'depth', 'full_reads_percent'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[58]:


df_train.columns


# In[59]:



#df_train['day'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%d").astype(int)
#df_train['mounth'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%m").astype(int)


# In[ ]:





# ## Correlation

# In[60]:


num_cols = ['views', 'depth', 'full_reads_percent', 
            'ctr',
            'hour', 'dow','mounth', #'day', 'weekend',
            'Nauthors', #'Nauthors_upd', 
            #'Ntags',
           ]


# In[61]:


f, ax = plt.subplots(figsize=(30, 16))
ax = sns.heatmap(df_train[num_cols].corr(), annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black',)


# In[62]:


f, ax = plt.subplots(figsize=(30, 16))
ax = sns.heatmap(df_train[num_cols].corr(method='spearman'), annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black')


# In[63]:


sns.pairplot(df_train[num_cols])


# In[ ]:





# In[ ]:





# In[ ]:




