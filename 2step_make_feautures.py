#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle as pkl

import numpy as np
import pandas as pd
from itertools import product

import category_encoders as ce
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from ast import literal_eval

from tqdm import tqdm
tqdm.pandas()


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


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'))#, index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test.csv'))#, index_col= 0)

df_train['publish_date'] = pd.to_datetime(df_train['publish_date'])
df_test['publish_date']  = pd.to_datetime(df_test['publish_date'])


# In[5]:


df_train.shape, df_test.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Очистка датасета

# этих категорий нет в тесте, а в трейне на них приходится всего 3 записи. они явно лишние.
# 
# уберем статьи раньше минимальной даты в тесте. для начала так, дальше можно будет поиграться.

# In[6]:


def clear_data(inp_df: pd.DataFrame, min_time: pd.Timestamp) -> pd.DataFrame:
    
    exclude_category = {'5e54e2089a7947f63a801742', '552e430f9a79475dd957f8b3', '5e54e22a9a7947f560081ea2' }
    inp_df = inp_df.query('category not in @exclude_category')
    print(f'shape after clean category {inp_df.shape}')
    
    inp_df = inp_df[inp_df.publish_date >= min_time]
    print(f'shape after min time {inp_df.shape}')
    
    
    if 'full_reads_percent' in inp_df.columns:
        inp_df = inp_df.query('full_reads_percent < 100')
        print(f'shape after frp time {inp_df.shape}')
                              
    
    return inp_df


# In[7]:


#min_test_time = df_test['publish_date'].min()
#min_test_time = df_test['publish_date'].nsmallest(2).iloc[-1]
min_test_time = pd.Timestamp('2022-01-01')

df_train = clear_data(df_train, min_test_time)
#df_test  = clear_data(df_test,  min_test_time)


# In[ ]:





# In[ ]:




exclude_category = {'5e54e2089a7947f63a801742', '552e430f9a79475dd957f8b3', '5e54e22a9a7947f560081ea2' }
df_train = df_train.query('category not in @exclude_category')
df_train.shape#min_time = pd.Timestamp('2021-05-17')
min_time = df_test['publish_date'].min()

df_train = df_train[df_train.publish_date > min_time]
df_train.shape
# In[ ]:





# In[ ]:





# In[ ]:





# ## title

# In[ ]:





# ## publish_date

# In[8]:


def publish_date_features(inp_df: pd.DataFrame) -> pd.DataFrame:
    
    inp_df['m_d'] = inp_df['publish_date'].dt.date

    inp_df['hour'] = inp_df['publish_date'].dt.hour
    inp_df['dow']  = inp_df['publish_date'].dt.dayofweek
    #Monday=0, Sunday=6
    #inp_df['weekend'] = (inp_df.dow >= 4).astype(int) # 5
    #inp_df['holidays']
    inp_df['day']    = pd.to_datetime(inp_df['publish_date']).dt.strftime("%d").astype(int)
    inp_df['mounth'] = pd.to_datetime(inp_df['publish_date']).dt.strftime("%m").astype(int)
    
    
    return inp_df

df_train['m_d'] = df_train['publish_date'].dt.date

df_train['hour'] = df_train['publish_date'].dt.hour
df_train['dow']  = df_train['publish_date'].dt.dayofweek
#Monday=0, Sunday=6
df_train['weekend'] = (df_train.dow >= 4).astype(int) # 5
#df_train['holidays']
df_train['day']    = pd.to_datetime(df_train['publish_date']).dt.strftime("%d").astype(int)
df_train['mounth'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%m").astype(int)df_test['m_d'] = df_test['publish_date'].dt.date

df_test['hour'] = df_test['publish_date'].dt.hour
df_test['dow']  = df_test['publish_date'].dt.dayofweek
#Monday=0, Sunday=6
df_test['weekend'] = (df_test.dow >= 4).astype(int) # 5
#df_train['holidays']
df_test['day']    = pd.to_datetime(df_test['publish_date']).dt.strftime("%d").astype(int)
df_test['mounth'] = pd.to_datetime(df_test['publish_date']).dt.strftime("%m").astype(int)
# In[ ]:





# In[9]:


df_train = publish_date_features(df_train)
df_test  = publish_date_features(df_test)


# In[ ]:





# In[10]:


df_train.sort_values(by='m_d').m_d.diff().value_counts()


# In[ ]:





# In[ ]:





# In[11]:


def create_daily_stats(inp_df: pd.DataFrame, max_lags: int = 7) -> pd.DataFrame:
    
    ret_df = inp_df.sort_values(by='m_d').groupby('m_d')[['m_d', 'views', 'depth', 'full_reads_percent']].agg(['min', 'max', 'mean', 'std']).copy()
    new_cols = ['views_min', 'views_max', 'views_mean', 'views_std',
                'depth_min', 'depth_max', 'depth_mean', 'depth_std',
                'frp_min',   'frp_max',   'frp_mean',   'frp_std',
               ]
    ret_df.columns = new_cols
    ret_df = ret_df.reset_index()
    #??????? only std
    #ret_df.isnull().sum() > 0
    ret_df.fillna(0, inplace = True)
    
    
    for col, lag in  product(new_cols, list(range(max_lags))):
        ret_df[f'{col}_lag{lag+1}'] = ret_df[col].shift(lag+1)
        #????fillna
        #ret_df[f'{col}_lag{lag+1}'].fillna('mean', inplace = True)
    
    return ret_df


# In[12]:


daily_stats = create_daily_stats(df_train)


# In[13]:


daily_stats.to_csv(os.path.join(DIR_DATA, 'dayly_stats.csv'), index = False)


# In[14]:


#daily_stats


# In[ ]:





# In[15]:


def add_daily_stats(inp_df:pd.DataFrame) -> pd.DataFrame:
    
    #ret_df = inp_df.merge(daily_stats, on = 'm_d', validate = 'many_to_one')
    ret_df = inp_df.merge(daily_stats, on = 'm_d', how = 'left', validate = 'many_to_one')
    
    
    return ret_df


# In[16]:


df_train.shape, daily_stats.shape


# In[17]:


df_train = add_daily_stats(df_train)


# In[18]:


df_train.shape


# In[19]:


df_test.shape


# In[20]:


df_test = add_daily_stats(df_test)


# In[21]:


df_test.shape

new_cols = ['views_min', 'views_max', 'views_mean', 'views_std',
            'depth_min', 'depth_max', 'depth_mean', 'depth_std',
            'frp_min',   'frp_max',   'frp_mean',   'frp_std',
           ]
#df_train[new_cols].isnull().sum()
# In[22]:


df_test[['views_min', 'views_max', 'views_mean', 'views_std',
            'depth_min', 'depth_max', 'depth_mean', 'depth_std',
            'frp_min',   'frp_max',   'frp_mean',   'frp_std']].isnull().sum()


# In[ ]:





# In[23]:


#df_train.drop('publish_date', axis = 1, inplace = True)
#df_test.drop('publish_date', axis = 1, inplace = True)


# In[ ]:





# ## session

# In[ ]:





# ## authors

# авторы считываются как строки, а не как массив строк. исправим.

# In[24]:


df_train['authors']  = df_train.authors.apply(lambda x: literal_eval(x))
df_test['authors']   = df_test.authors.apply( lambda x: literal_eval(x))


# In[25]:


df_train['authors'] = df_train['authors'].apply(lambda x: x if len(x) > 0 else ['without_author'])
df_test['authors']  = df_test['authors'].apply( lambda x: x if len(x) > 0 else ['without_author'])


# In[ ]:





# In[26]:


all_authors = set()
for el in df_train.authors.values:
    if len (el) == 0:
        continue
    if len(el) == 1:
        all_authors.add(el[0])
        continue
        
    for author in el:
        all_authors.add(author)


# In[27]:


#for el in df_train.loc[:5, ['document_id', 'authors']].values:
#    print(el)


# Все статьи автора (с учетом совместных)

# In[28]:


auth_doc_id = {el: [] for el in all_authors}

for el in tqdm(df_train.loc[:, ['document_id', 'authors']].values):
    for athr in range(len(el[1])):
        auth_doc_id[el[1][athr]].append(el[0])


# In[29]:


with open(os.path.join(DIR_DATA, 'authors_all.pkl'), 'wb') as pkl_file:
    pkl.dump(auth_doc_id, pkl_file)


# Статьи только автора (в одиночку)

# In[30]:


auth_doc_id_alone = {el: [] for el in all_authors}

for el in tqdm(df_train.loc[:, ['document_id', 'authors']].values):
    if len(el[1]) == 1:
        auth_doc_id_alone[el[1][0]].append(el[0])


# In[31]:


with open(os.path.join(DIR_DATA, 'authors_alone.pkl'), 'wb') as pkl_file:
    pkl.dump(auth_doc_id_alone, pkl_file)


# In[ ]:





# In[ ]:





# ## ctr

# In[ ]:





# ## category

# In[32]:


def create_daily_stats_by_category(inp_df: pd.DataFrame, max_lags: int = 7) -> pd.DataFrame:
    
    ret_df = inp_df[['publish_date', 'm_d', 'category', 'views', 'depth', 'full_reads_percent']].copy()
    new_cols = ['cat_views_min', 'cat_views_max', 'cat_views_mean', 'cat_views_std',
                'cat_depth_min', 'cat_depth_max', 'cat_depth_mean', 'cat_depth_std',
                'cat_frp_min',   'cat_frp_max',   'cat_frp_mean',   'cat_frp_std',
               ]
    
    ret_df.sort_values(by=['publish_date'], inplace = True)
    ret_df = ret_df.groupby(['category', 'm_d'])['views', 'depth', 'full_reads_percent'].agg(('min', 'max', 'mean', 'std'))
        
    ret_df.columns = new_cols
    ret_df = ret_df.reset_index()
    #??????? only std
    #ret_df.isnull().sum() > 0
    ret_df.fillna(0, inplace = True)
    
    
    for col, lag in  product(new_cols, list(range(max_lags))):
        ret_df[f'{col}_lag{lag+1}'] = ret_df[col].shift(lag+1)
        #????fillna
        #ret_df[f'{col}_lag{lag+1}'].fillna('mean', inplace = True)
        
    return ret_df


# In[33]:


daily_stats_category = create_daily_stats_by_category(df_train)

daily_stats_category = df_train[['document_id', 'publish_date', 'm_d', 'category', 'views', 'depth', 'full_reads_percent']].copy()
daily_stats_category.sort_values(by=['category', 'publish_date'], inplace = True)

daily_stats_category = daily_stats_category.groupby(['category', 'm_d'])['views', 'depth', 'full_reads_percent'].agg(('min', 'max', 'mean', 'std'))
daily_stats_category.columns = new_cat

daily_stats_category.reset_index(inplace = True)
# In[34]:


#daily_stats_category.head(10)


# In[35]:


#daily_stats_category.groupby('category').agg('size')


# In[36]:


daily_stats_category.to_csv(os.path.join(DIR_DATA, 'daily_stats_category.csv'), index = False)


# In[ ]:





# In[ ]:





# In[37]:


def add_daily_stats_category(inp_df:pd.DataFrame) -> pd.DataFrame:
    
    ret_df = inp_df.merge(daily_stats_category, on = ['category', 'm_d'], how = 'left', validate = 'many_to_one')
    
    
    return ret_df


# In[38]:


df_train.shape, daily_stats_category.shape


# In[39]:


df_train = add_daily_stats_category(df_train)


# In[40]:


df_train.shape


# In[41]:


df_test = add_daily_stats_category(df_test)


# In[42]:


df_test[['cat_views_min', 'cat_views_max', 'cat_views_mean', 'cat_views_std',
                'cat_depth_min', 'cat_depth_max', 'cat_depth_mean', 'cat_depth_std',
                'cat_frp_min',   'cat_frp_max',   'cat_frp_mean',   'cat_frp_std',]].isnull().sum()


# In[ ]:





# In[ ]:





# ## tags

# In[43]:


df_train['tags']  = df_train.tags.apply(lambda x: literal_eval(x))
df_test['tags']   = df_test.tags.apply( lambda x: literal_eval(x))


# In[ ]:





# In[ ]:





# разделяем категориальные и числовые признаки   
# числовые нормализуем

# In[44]:


#df_train.columns.to_list()


# In[45]:


cat_cols = ['hour', 'dow', 'day', 'mounth',
           ]

num_cols = [el for el in df_test.columns.to_list() if el not in cat_cols]
num_cols = [el for el in num_cols if el not in ['document_id', 'title', 'publish_date', 'm_d', 'session', 'authors', 'category', 'tags', 
                                                
                                                'views_min', 'views_max', 'views_mean', 'views_std',   # not interested in the current day
                                                'depth_min', 'depth_max', 'depth_mean', 'depth_std',   # not interested in the current day
                                                'frp_min', 'frp_max', 'frp_mean', 'frp_std',           # not interested in the current day
                                                
                                                'cat_views_min', 'cat_views_max', 'cat_views_mean', 'cat_views_std',  # not interested in the current day
                                                'cat_depth_min', 'cat_depth_max', 'cat_depth_mean', 'cat_depth_std',  # not interested in the current day
                                                'cat_frp_min',   'cat_frp_max',   'cat_frp_mean',   'cat_frp_std',    # not interested in the current day
                                                
                                                'views', 'depth', 'full_reads_percent']]


# In[46]:


#num_cols


# In[47]:


for el in num_cols:
    if el not in df_train.columns or el not in df_test.columns:
        print(el)
        
for el in cat_cols:
    if el not in df_train.columns or el not in df_test.columns:
        print(el)


# ## normalize

# In[48]:


#scaler = preprocessing.MinMaxScaler()   #Transform features by scaling each feature to a given range.
#scaler = preprocessing.Normalizer()     #Normalize samples individually to unit norm.
scaler = preprocessing.StandardScaler()  #Standardize features by removing the mean and scaling to unit variance.

scaler.fit(df_train[num_cols])


# In[49]:


#df_train[num_cols].head(5)


# In[50]:


#df_test[num_cols].head(5)


# In[51]:


df_train[num_cols] = scaler.transform(df_train[num_cols])
df_test[num_cols]  = scaler.transform(df_test[num_cols])


# In[52]:


#df_train[num_cols].head(5)


# In[53]:


#df_test[num_cols].head(5)


# In[ ]:





# Добавляем эмбединги

# In[54]:


# sberbank-ai/sbert_large_mt_nlu_ru       1024  1.71Gb
# DeepPavlov/rubert-base-cased-sentence   768   0.7Gb
# DeepPavlov/rubert-base-cased-conversational  768
# DeepPavlov/rubert-base-cased            768
# sberbank-ai/sbert_large_nlu_ru          1024  1.71Gb

MODEL_FOLDER = 'sbert_large_mt_nlu_ru'
MAX_LENGTH = 24

def add_ttle_embeding(inp_df: pd.DataFrame) -> pd.DataFrame:
    
    pass    
    
# In[55]:


emb_train = pd.read_csv(os.path.join(DIR_DATA, f'ttl_emb_train_{MODEL_FOLDER}_{MAX_LENGTH}.csv'))
#emb_train.drop(['document_id', 'title'], axis = 1 , inplace = True)
emb_train.drop(['title'], axis = 1 , inplace = True)

df_train = df_train.merge(emb_train, on = 'document_id', validate = 'one_to_one')
df_train.shape, emb_train.shape


# In[56]:


emb_test = pd.read_csv(os.path.join(DIR_DATA, f'ttl_emb_test_{MODEL_FOLDER}_{MAX_LENGTH}.csv'))
#emb_test.drop(['document_id', 'title'], axis = 1 , inplace = True)
emb_test.drop(['title'], axis = 1 , inplace = True)

df_test = df_test.merge(emb_test, on = 'document_id', validate = 'one_to_one')
df_test.shape, emb_test.shape


# In[57]:


num_cols = num_cols + list(emb_train.columns)


# In[58]:


if 'document_id' in num_cols:
    num_cols.remove('document_id')


# In[ ]:





# ## train_test_split

# вероятно лучше разделять до нормализации и категориальных энкодеров, что бы значения из валидационной выборки не были в учтены в тесте   
# однако, на первой итерации устроит и разбиение после всех преобразований

# In[59]:


x_train, x_val = train_test_split(df_train, stratify = df_train['category'], test_size = 0.2)
df_train.shape, x_train.shape, x_val.shape


# In[ ]:





# ## save

# In[60]:


df_test.shape


# In[61]:


x_train.to_csv(os.path.join(DIR_DATA,  'x_train.csv'))
x_val.to_csv(os.path.join(DIR_DATA,    'x_val.csv'))
df_test.to_csv(os.path.join( DIR_DATA, 'test_upd.csv'))


# In[62]:


with open(os.path.join(DIR_DATA, 'num_columns.pkl'), 'wb') as pickle_file:
    pkl.dump(num_cols, pickle_file)


# In[63]:


with open(os.path.join(DIR_DATA, 'cat_columns.pkl'), 'wb') as pickle_file:
    pkl.dump(cat_cols, pickle_file)


# In[64]:


#df_test.columns


# In[ ]:





# In[ ]:




