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


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train_extended.csv'))#, index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test_extended.csv'))#, index_col= 0)

df_train['publish_date'] = pd.to_datetime(df_train['publish_date'])
df_test['publish_date']  = pd.to_datetime(df_test['publish_date'])


# In[5]:


df_train.shape, df_test.shape


# In[ ]:





# Имена признаков для удобства перебора будут представлены словарем   
# Формат:   
# {
# исходный признак/идея: {   
# только числовые признаки: [ ]   
# только категориальные признаки: [ ]   
# признаки, которые могу быть как числовыми так и категориальными: [ ]   
# }}

# In[6]:


clmns = {'document_id':{'num':  ['nimgs','text_len', ],   
                        'cat':  [],
                        'both': [],
                        },
        'title':         {'num':  [],   
                          'cat':  [],
                          'both': [],
                         }, 
        'publish_date': {'num':  [],   
                          'cat':  [],
                          'both': [],
                         },
         'authors': {'num':  [],   
                     'cat':  [],
                     'both': [],
                    },
         'ctr': {'num':  [],   
                 'cat':  [],
                 'both': [],
                },
         'category': {'num':  [],   
                      'cat':  [],
                      'both': [],
                     },
         'title': {'num':  [],   
                   'cat':  ['two_articles'],
                   'both': [],
                },
        }


# In[7]:


print(df_train.columns.values)


# In[ ]:





# ## Очистка датасета

# этих категорий нет в тесте, а в трейне на них приходится всего 3 записи. они явно лишние.
# 
# уберем статьи раньше минимальной даты в тесте. для начала так, дальше можно будет поиграться.

# In[8]:


def clear_data(inp_df: pd.DataFrame, min_time: pd.Timestamp) -> pd.DataFrame:
    
    exclude_category = {'5e54e2089a7947f63a801742', '552e430f9a79475dd957f8b3', '5e54e22a9a7947f560081ea2' }
    inp_df = inp_df.query('category not in @exclude_category')
    print(f'shape after clean category {inp_df.shape}')
    
    inp_df = inp_df[inp_df.publish_date >= min_time]
    print(f'shape after min time {inp_df.shape}')
    
    inp_df = inp_df.query('ctr != 6.096')
    print(f'shape after ctr {inp_df.shape}')
    
    if 'full_reads_percent' in inp_df.columns:
        inp_df = inp_df.query('full_reads_percent < 100')
        print(f'shape after frp time {inp_df.shape}')
                              
    
    Q1_v = inp_df['views'].quantile(0.25)
    Q3_v = inp_df['views'].quantile(0.75)
    IQR_v = Q3_v - Q1_v
    Q1_d = inp_df['depth'].quantile(0.25)
    Q3_d = inp_df['depth'].quantile(0.75)
    IQR_d = Q3_d - Q1_d
    Q1_f = inp_df['full_reads_percent'].quantile(0.25)
    Q3_f = inp_df['full_reads_percent'].quantile(0.75)
    IQR_f = Q3_f - Q1_f
    
    inp_df = inp_df.query('(@Q1_v - 1.5 * @IQR_v) <= views <= (@Q3_v + 1.5 * @IQR_v)')
    inp_df = inp_df.query('(@Q1_d - 1.5 * @IQR_d) <= depth <= (@Q3_d + 1.5 * @IQR_d)')
    inp_df = inp_df.query('(@Q1_f - 1.5 * @IQR_f) <= full_reads_percent <= (@Q3_f + 1.5 * @IQR_f)')
    
    return inp_df


# In[9]:


#min_test_time = df_test['publish_date'].min()
#min_test_time = df_test['publish_date'].nsmallest(2).iloc[-1]
min_test_time = pd.Timestamp('2022-01-01')

df_train = clear_data(df_train, min_test_time)
#df_test  = clear_data(df_test,  min_test_time)


# In[ ]:





# In[ ]:





# ## title

# In[10]:


def add_title_features(inp_df):
    
    # Прямая трансляция, Фоторепортаж, Фотогалерея, Видео, телеканале РБК, Инфографика endswith
    
    inp_df['ph_report']  = inp_df.true_title.apply(lambda x: 1 if x.endswith('Фоторепортаж') else 0)
    inp_df['ph_gallery'] = inp_df.true_title.apply(lambda x: 1 if x.endswith('Фотогалерея') else 0)
    inp_df['tv_prog'] = inp_df.true_title.apply(lambda x: 1 if x.endswith('телеканале РБК') else 0)
    inp_df['online'] = inp_df.true_title.apply(lambda x: 1 if x.endswith('Прямая трансляция') else 0)
    inp_df['video']  = inp_df.true_title.apply(lambda x: 1 if x.endswith('Видео') else 0)
    inp_df['infogr'] = inp_df.true_title.apply(lambda x: 1 if x.endswith('Инфографика') else 0)
    
    if 'video' not in clmns['title']['both']:
        clmns['title']['both'].extend(['ph_report', 'ph_gallery', 'tv_prog', 'online', 'video', 'infogr'])
        
    return inp_df


# In[11]:


df_train = add_title_features(df_train)
df_test = add_title_features(df_test)


# In[12]:


df_train.ph_report.sum(), df_train.ph_gallery.sum(), df_train.tv_prog.sum(), df_train.online.sum(), df_train.video.sum(), df_train.infogr.sum()


# In[13]:


df_test.ph_report.sum(), df_test.ph_gallery.sum(), df_test.tv_prog.sum(), df_test.online.sum(), df_test.video.sum(), df_test.infogr.sum()


# In[ ]:





# # publish date

# In[14]:


holidays = {pd.Timestamp('2022-01-01').date(), pd.Timestamp('2022-01-02').date(), pd.Timestamp('2022-01-03').date(),
            pd.Timestamp('2022-01-04').date(), pd.Timestamp('2022-01-05').date(), pd.Timestamp('2022-01-06').date(),  #NY
            pd.Timestamp('2022-01-07').date(), pd.Timestamp('2022-01-08').date(), pd.Timestamp('2022-01-08').date(),
            pd.Timestamp('2022-02-23').date(), # 23 feb
            pd.Timestamp('2022-03-06').date(), pd.Timestamp('2022-03-07').date(), pd.Timestamp('2022-03-08').date(), # 8 march
            pd.Timestamp('2022-05-02').date(), pd.Timestamp('2022-05-03').date(), # 1st may
            pd.Timestamp('2022-05-09').date(), pd.Timestamp('2022-05-10').date(),# 9 may
            pd.Timestamp('2022-06-12').date(), pd.Timestamp('2022-06-13').date(), # day of the russia
            pd.Timestamp('2022-11-04').date()
           }

day_before_holiday = {pd.Timestamp('2021-12-31').date(), pd.Timestamp('2022-02-22').date(), pd.Timestamp('2022-03-05').date(),
                      pd.Timestamp('2022-02-23').date(),
                      pd.Timestamp('2022-04-29').date(), pd.Timestamp('2022-05-04').date(), 
                      pd.Timestamp('2022-05-05').date(), pd.Timestamp('2022-05-06').date(),
                      pd.Timestamp('2022-11-03').date(),
                      #pd.Timestamp('2022-12-03').date(),
                      #pd.Timestamp('2022-11-03').date(),
                     }
day_after_holiday = {pd.Timestamp('2022-01-10').date(), pd.Timestamp('2022-02-24').date(), pd.Timestamp('2022-03-09').date(), 
                     pd.Timestamp('2022-06-14').date(), pd.Timestamp('2022-05-11').date(),
                    }


# In[15]:


border = pd.Timestamp('2022-04-08').date()


# In[16]:


def publish_date_features(inp_df: pd.DataFrame) -> pd.DataFrame:
    
    inp_df['m_d'] = inp_df['publish_date'].dt.date

    inp_df['hour'] = inp_df['publish_date'].dt.hour
    inp_df['dow']  = inp_df['publish_date'].dt.dayofweek
    #Monday=0, Sunday=6
    #inp_df['weekend'] = (inp_df.dow >= 4).astype(int) # 5
    #inp_df['holidays']
    inp_df['day']    = pd.to_datetime(inp_df['publish_date']).dt.strftime("%d").astype(int)
    inp_df['mounth'] = pd.to_datetime(inp_df['publish_date']).dt.strftime("%m").astype(int)
    
    
    inp_df['holiday'] = inp_df.m_d.apply(lambda x: 1 if x in holidays else 0)
    inp_df['day_before_holiday'] = inp_df.m_d.apply(lambda x: 1 if x in day_before_holiday else 0)
    inp_df['day_after_holiday'] = inp_df.m_d.apply(lambda x: 1 if x in day_after_holiday else 0)
    
    inp_df['distrib_brdr'] = inp_df.m_d.apply(lambda x: 1 if x < border else 0)
    
    if 'hour' not in clmns['publish_date']['both']:
        clmns['publish_date']['both'].extend(['hour', 'dow', 'day', 'mounth'])#, 'distrib_brdr'])
        
    #if 'holiday' not in clmns['publish_date']['cat']:
    #    clmns['publish_date']['cat'].extend(['holiday', 'day_before_holiday', 'day_after_holiday',]) 
    
    if 'holiday' not in clmns['publish_date']['both']:
        clmns['publish_date']['both'].extend(['holiday', 'day_before_holiday', 'day_after_holiday', 'distrib_brdr']) 
    
    return inp_df


# In[17]:


print('before ', df_train.shape, df_test.shape)
df_train = publish_date_features(df_train)
df_test  = publish_date_features(df_test)
print('after  ', df_train.shape, df_test.shape)


# In[18]:


print(sum(df_train.holiday), sum(df_train.day_before_holiday), sum(df_train.day_after_holiday), )
print(sum(df_test.holiday), sum(df_test.day_before_holiday), sum(df_test.day_after_holiday), )


# In[ ]:





# Рассчитаем дневные статистики + лаги за 7 дней

# In[19]:


df_train.sort_values(by='m_d').m_d.diff().value_counts()


# In[20]:


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


# In[21]:


daily_stats = create_daily_stats(df_train)
daily_stats.to_csv(os.path.join(DIR_DATA, 'dayly_stats.csv'), index = False)


# In[ ]:





# Добавим их к датасетам

# In[22]:


def add_daily_stats(inp_df:pd.DataFrame) -> pd.DataFrame:
    
    #ret_df = inp_df.merge(daily_stats, on = 'm_d', validate = 'many_to_one')
    ret_df = inp_df.merge(daily_stats, on = 'm_d', how = 'left', validate = 'many_to_one')
    
    if 'views_min' not in clmns['publish_date']['num']:
        clmns['publish_date']['num'].extend(daily_stats.columns[1:])
    
    return ret_df


# In[23]:


print('before ', df_train.shape, df_test.shape, 'add ', daily_stats.shape)
df_train = add_daily_stats(df_train)
df_test  = add_daily_stats(df_test)
print('after  ', df_train.shape, df_test.shape)


# Проверим на пропуски в тесте

# In[24]:


df_test[['views_min', 'views_max', 'views_mean', 'views_std',
            'depth_min', 'depth_max', 'depth_mean', 'depth_std',
            'frp_min',   'frp_max',   'frp_mean',   'frp_std']].isnull().sum()


# да, на начальныз лагах есть пропуски.    
# заменять будем уже при подборе и построении моделей   

# In[ ]:





# In[ ]:





# In[ ]:





# ## session

# In[ ]:





# ## authors

# Авторы считываются как строки, а не как массив строк. исправим.

# In[25]:


df_train['authors']  = df_train.authors.apply(lambda x: literal_eval(x))
df_test['authors']   = df_test.authors.apply( lambda x: literal_eval(x))

# пустое поле автора заменим на значение, что автор не указан
df_train['authors'] = df_train['authors'].apply(lambda x: x if len(x) > 0 else ['without_author'])
df_test['authors']  = df_test['authors'].apply( lambda x: x if len(x) > 0 else ['without_author'])


# In[26]:


df_train['Nauthors'] = df_train.authors.apply(lambda x: len(x))
df_test['Nauthors']  = df_test.authors.apply(lambda x: len(x))


# In[27]:


clmns['authors']['num'].extend(['Nauthors'])


# выделяем всех авторов в трейне

# In[28]:


all_authors = set()
for el in df_train.authors.values:
    if len (el) == 0:
        continue
    if len(el) == 1:
        all_authors.add(el[0])
        continue
        
    for author in el:
        all_authors.add(author)


# проверяем на наличия авторов из теста

# In[29]:


test_authors = set()
for el in df_test.authors.values:
    if len (el) == 0:
        continue
    if len(el) == 1:
        test_authors.add(el[0])
        continue
        
    for author in el:
        test_authors.add(author)

for el in test_authors:
    if el not in all_authors:
        print(el)


# 2х авторов нет в трейне.   
# предположительно заменим их статистики средними.

# In[ ]:





# Все статьи автора (с учетом совместных)

# In[30]:


auth_doc_id = {el: [] for el in all_authors}

for el in tqdm(df_train.loc[:, ['document_id', 'authors']].values):
    for athr in range(len(el[1])):
        auth_doc_id[el[1][athr]].append(el[0])
        
with open(os.path.join(DIR_DATA, 'authors_all.pkl'), 'wb') as pkl_file:
    pkl.dump(auth_doc_id, pkl_file)


# Статьи только автора (в одиночку)(пока не применяется)

# In[31]:


auth_doc_id_alone = {el: [] for el in all_authors}

for el in tqdm(df_train.loc[:, ['document_id', 'authors']].values):
    if len(el[1]) == 1:
        auth_doc_id_alone[el[1][0]].append(el[0])
        
with open(os.path.join(DIR_DATA, 'authors_alone.pkl'), 'wb') as pkl_file:
    pkl.dump(auth_doc_id_alone, pkl_file)


# In[ ]:





# Соберем статистику по авторам (с учетом совместных)

# In[32]:


author_columns = ['author', 'author_size', 'v_auth_min', 'v_auth_max', 'v_auth_mean', 'v_auth_std', 'd_auth_min',
                  'd_auth_max', 'd_auth_mean', 'd_auth_std', 'f_auth_min', 'f_auth_max', 'f_auth_mean', 'f_auth_std',
                 ]

author_group_columns = ['author_size', 
                        'v_auth_min', 'v_auth_max', 'v_auth_mean', 'v_auth_std',
                        'author_size2',
                        'd_auth_min', 'd_auth_max', 'd_auth_mean', 'd_auth_std',
                        'author_size3',
                        'f_auth_min', 'f_auth_max', 'f_auth_mean', 'f_auth_std',
                   ]


# In[33]:


df_author = pd.DataFrame(columns = author_columns)
df_author.author = list(all_authors)

for el in tqdm(all_authors):
    # определяем статьи текущего автора
    df_train['cur_author'] = df_train.authors.apply(lambda x: 1 if el in x else 0)
    
    # собираем статистики текущего автора
    tmp = df_train.groupby('cur_author')[['views', 'depth', 'full_reads_percent']].agg(['size', 'min', 'max', 'mean', 'std'])
    tmp.columns = author_group_columns
    tmp.reset_index(inplace = True)
    tmp.drop(['author_size2', 'author_size3'], axis = 1, inplace = True)
    
    # сохраняем полученные статистики в DataFrame
    df_author.loc[df_author.query('author == @el').index, author_columns[1:]] = tmp.query('cur_author == 1')[tmp.columns[1:]].values[0]
    
    
    
# для 2х неизвестных авторов из теста добавим их средними
# правильнее бы добавить в функцию добавления статистки, а не в сам DataFrame
# однако на данном этапе такой вариант нас более чем устроит
#'5a2511349a794727e3fa3d20'
#'57f766ae9a79479bfcfa0133'
df_author.loc['mean'] = df_author.mean()
df_author.loc['mean2'] = df_author.loc['mean']

df_author.loc['mean', ['author']] = '5a2511349a794727e3fa3d20'
df_author.loc['mean2', ['author']] = '57f766ae9a79479bfcfa0133'


# In[34]:


df_author.to_csv(os.path.join(DIR_DATA, 'author_together.csv'), index = False)


# In[35]:


#df_author.tail()


# In[ ]:





# Добавляем статистики по авторам в датасеты

# In[36]:


def add_author_statistics(inp_df):
    
    if len(inp_df[0]) == 0:  # заменяли на without_author так что не может быть
        print(inp_Df)
    elif len(inp_df[0]) == 1:
        return df_author.loc[df_author.author == inp_df[0][0], 
                              author_columns[2:]
                            ].values[0]
    else:
        ret_np  = np.zeros(shape = (len(author_columns[2:]),) )
        divisor = len(inp_df[0])
        
        # если авторо больше одного будем выбират средние/мин/макс наченяи среди них
        for el in inp_df[0]:
            if el in df_author.author.values:
                tmp = df_author[df_author.author == el]
                ret_np = [ret_np[0]  + tmp.v_auth_min.values[0],
                          ret_np[1]  + tmp.v_auth_max.values[0],
                          ret_np[2]  + tmp.v_auth_mean.values[0],
                          ret_np[3]  + tmp.v_auth_std.values[0],
                          ret_np[4]  + tmp.d_auth_min.values[0],
                          ret_np[5]  + tmp.d_auth_max.values[0],
                          ret_np[6]  + tmp.d_auth_mean.values[0],
                          ret_np[7]  + tmp.d_auth_std.values[0],
                          ret_np[8]  + tmp.f_auth_min.values[0],
                          ret_np[9]  + tmp.f_auth_max.values[0],
                          ret_np[10] + tmp.f_auth_mean.values[0],
                          ret_np[11] + tmp.f_auth_std.values[0]
                         ]
            else: # aouthor in test out from train
                ret_np = [ret_np[0]  + 0,
                          ret_np[1]  + 0,
                          ret_np[2]  + 0,
                          ret_np[3]  + 0,
                          ret_np[4]  + 0,
                          ret_np[5]  + 0,
                          ret_np[6]  + 0,
                          ret_np[7]  + 0,
                          ret_np[8]  + 0,
                          ret_np[9]  + 0,
                          ret_np[10] + 0,
                          ret_np[11] + 0
                         ]
                
        #№ пока только среднее
        ret_np = [ret_np[0]  / divisor,   # v_auth_min OR MIN
                  ret_np[1]  / divisor,   # v_auth_max OR MAX
                  ret_np[2]  / divisor,   # v_auth_mean
                  ret_np[3]  / divisor,   # v_auth_std
                  ret_np[4]  / divisor,   # d_auth_min OR MIN
                  ret_np[5]  / divisor,   # d_auth_max OR MAX
                  ret_np[6]  / divisor,   # d_auth_mean
                  ret_np[7]  / divisor,   # d_auth_std
                  ret_np[8]  / divisor,   # f_auth_min OR MIN
                  ret_np[9]  / divisor,   # f_auth_max OR MAX
                  ret_np[10] / divisor,   # f_auth_mean
                  ret_np[11] / divisor,   # f_auth_std
                 ]
        
    return ret_np


# In[37]:


# кроме полей author / author_size

print('before', df_train.shape, df_test.shape)
author_stats_train = df_train[['authors']].progress_apply(add_author_statistics, axis = 1)
author_stats_test  = df_test[ ['authors']].progress_apply(add_author_statistics, axis = 1)

#df_train = pd.concat([df_train, pd.DataFrame(author_stats_train.to_list(), columns = author_columns[2:])], axis = 1)
#df_test  = pd.concat([df_test , pd.DataFrame(author_stats_test.to_list(),  columns = author_columns[2:])], axis = 1)
print('after', df_train.shape, df_test.shape)


# In[38]:


#clmns['authors']['num'].extend(author_columns[2:])


# In[ ]:





# In[ ]:





# In[ ]:





# ## ctr

# In[39]:


def add_ctr_features(inp_df):
    
    inp_df['spec_event_1'] = inp_df.ctr.apply(lambda x: 1 if x == 6.096 else 0)
    
    if 'spec_event_1' not in clmns['ctr']['both']:
        clmns['ctr']['both'].extend(['spec_event_1']) 
                                       
    return inp_df


# In[40]:


#print('before ', df_train.shape, df_test.shape)
#df_train = add_ctr_features(df_train)
#df_test  = add_ctr_features(df_test)
#print('after  ', df_train.shape, df_test.shape)


# In[ ]:





# In[ ]:





# ## category

# Собираем статистики по категориям

# In[41]:


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


# In[42]:


daily_stats_category = create_daily_stats_by_category(df_train)
daily_stats_category.to_csv(os.path.join(DIR_DATA, 'daily_stats_category.csv'), index = False)


# In[ ]:





# Добавляем статистики по категориям в датасеты

# In[43]:


def add_daily_stats_category(inp_df:pd.DataFrame) -> pd.DataFrame:
    
    ret_df = inp_df.merge(daily_stats_category, on = ['category', 'm_d'], how = 'left', validate = 'many_to_one')
    
    
    return ret_df


# In[44]:


print('before ', df_train.shape, df_test.shape, 'add ', daily_stats_category.shape)
df_train = add_daily_stats_category(df_train)
df_test = add_daily_stats_category(df_test)
print('after  ', df_train.shape, df_test.shape, )


# In[45]:


clmns['category']['num'].extend(daily_stats_category.columns[2:])


# Проверяем, что все данные есть в тесте

# In[46]:


#df_test[['cat_views_min', 'cat_views_max', 'cat_views_mean', 'cat_views_std',
#                'cat_depth_min', 'cat_depth_max', 'cat_depth_mean', 'cat_depth_std',
#               'cat_frp_min',   'cat_frp_max',   'cat_frp_mean',   'cat_frp_std',]].isnull().sum()
df_test[daily_stats_category.columns[2:]].isnull().sum()


# Значения в признаках с лагами могут отсутствовать

# In[ ]:





# In[ ]:





# In[ ]:





# ## tags

# In[47]:


df_train['tags']  = df_train.tags.apply(lambda x: literal_eval(x))
df_test['tags']   = df_test.tags.apply( lambda x: literal_eval(x))


# In[ ]:





# In[ ]:





# ## Предобработка признаков в датасетах

# выделяем числовые признаки для нормализации
# исключаем из нормализации категориальные признаки
# признаки, которые могут быть как категориальными, так и числовыми
# так же нормализуем, т.к. после нормализации их 'количество категорий' не меняется
cat_cols = []
for el in clmns.keys():
    cat_cols.extend(clmns[el]['cat'])


num_cols = [el for el in df_test.columns.to_list() if el not in cat_cols]
num_cols = [el for el in num_cols if el not in ['document_id', 'title', 'publish_date', 'm_d', 'session', 'authors', 'category', 'tags', 
                                                
                                                'true_category', 'true_title', 'overview', 
                                                
                                                'views_min', 'views_max', 'views_mean', 'views_std',   # not interested in the current day
                                                'depth_min', 'depth_max', 'depth_mean', 'depth_std',   # not interested in the current day
                                                'frp_min', 'frp_max', 'frp_mean', 'frp_std',           # not interested in the current day
                                                
                                                'cat_views_min', 'cat_views_max', 'cat_views_mean', 'cat_views_std',  # not interested in the current day
                                                'cat_depth_min', 'cat_depth_max', 'cat_depth_mean', 'cat_depth_std',  # not interested in the current day
                                                'cat_frp_min',   'cat_frp_max',   'cat_frp_mean',   'cat_frp_std',    # not interested in the current day
                                                
                                                'views', 'depth', 'full_reads_percent']]for el in num_cols:
    if el not in df_train.columns or el not in df_test.columns:
        print(el)
        
for el in cat_cols:
    if el not in df_train.columns or el not in df_test.columns:
        print(el)
# In[48]:


cat_cols = []
num_cols = []

for el in clmns.keys():
    cat_cols.extend(clmns[el]['cat'])
    num_cols.extend(clmns[el]['num'])
    if len(clmns[el]['both']) != 0:
        print(clmns[el]['both'])


# In[49]:


num_cols.extend(['hour', 'mounth', 'dow', ])
cat_cols.extend([ 'ph_report', 'ph_gallery', 'tv_prog', 'online', 'video', 'infogr',
                  'holiday', 'day_before_holiday', 'day_after_holiday', 'distrib_brdr',
                  #'spec_event_1'
                ])


# нормализуем

# In[50]:


#scaler = preprocessing.MinMaxScaler()   #Transform features by scaling each feature to a given range.
#scaler = preprocessing.Normalizer()     #Normalize samples individually to unit norm.
scaler = preprocessing.StandardScaler()  #Standardize features by removing the mean and scaling to unit variance.

scaler.fit(df_train[num_cols])

df_train[num_cols] = scaler.transform(df_train[num_cols])
df_test[num_cols]  = scaler.transform(df_test[num_cols])


# In[ ]:





# In[ ]:





# In[ ]:





# Добавляем эмбединги

# In[51]:


# sberbank-ai/sbert_large_mt_nlu_ru       1024  1.71Gb
# DeepPavlov/rubert-base-cased-sentence   768   0.7Gb
# DeepPavlov/rubert-base-cased-conversational  768
# DeepPavlov/rubert-base-cased            768
# sberbank-ai/sbert_large_nlu_ru          1024  1.71Gb

#MODEL_FOLDER = 'rubert-base-cased-sentence'
MODEL_FOLDER = 'sbert_large_mt_nlu_ru'
MAX_LENGTH = 24
PCA_COMPONENTS = 64

def add_ttle_embeding(inp_df: pd.DataFrame) -> pd.DataFrame:
    
    pass    
    
# In[52]:


emb_train = pd.read_csv(os.path.join(DIR_DATA, f'ttl_cln_emb_train_{MODEL_FOLDER}_{MAX_LENGTH}_pca{PCA_COMPONENTS}.csv'))
#emb_train.drop(['document_id', 'title'], axis = 1 , inplace = True)
emb_train.drop(['true_title'], axis = 1 , inplace = True)

df_train = df_train.merge(emb_train, on = 'document_id', validate = 'one_to_one')
df_train.shape, emb_train.shape


# In[53]:


emb_test = pd.read_csv(os.path.join(DIR_DATA, f'ttl_cln_emb_test_{MODEL_FOLDER}_{MAX_LENGTH}_pca{PCA_COMPONENTS}.csv'))
#emb_test.drop(['document_id', 'title'], axis = 1 , inplace = True)
emb_test.drop(['true_title'], axis = 1 , inplace = True)

df_test = df_test.merge(emb_test, on = 'document_id', validate = 'one_to_one')
df_test.shape, emb_test.shape


# In[54]:


num_cols = num_cols + list(emb_train.columns)


# In[55]:


if 'document_id' in num_cols:
    num_cols.remove('document_id')


# In[56]:


clmns['title']['num'].extend(emb_train.columns[1:])


# In[ ]:





# ## train_test_split

# вероятно лучше разделять до нормализации и категориальных энкодеров, что бы значения из валидационной выборки не были в учтены в тесте   
# однако, на первой итерации устроит и разбиение после всех преобразований

# In[57]:


x_train, x_val = train_test_split(df_train, stratify = df_train['category'], test_size = 0.2)
df_train.shape, x_train.shape, x_val.shape


# In[ ]:





# ## save

# In[58]:


df_test.shape, x_train.shape, x_val.shape, df_test.shape


# In[59]:


df_train.to_csv(os.path.join( DIR_DATA, 'train_upd.csv'))
df_test.to_csv(os.path.join( DIR_DATA,  'test_upd.csv'))
x_train.to_csv(os.path.join(DIR_DATA,   'x_train.csv'))
x_val.to_csv(os.path.join(DIR_DATA,     'x_val.csv'))


# In[60]:


with open(os.path.join(DIR_DATA, 'num_columns.pkl'), 'wb') as pickle_file:
    pkl.dump(num_cols, pickle_file)


# In[61]:


with open(os.path.join(DIR_DATA, 'cat_columns.pkl'), 'wb') as pickle_file:
    pkl.dump(cat_cols, pickle_file)


# In[62]:


with open(os.path.join(DIR_DATA, 'clmns.pkl'), 'wb') as pickle_file:
    pkl.dump(clmns, pickle_file)


# In[ ]:





# In[ ]:





# In[63]:


cat_cols


# In[64]:


print(num_cols)


# In[ ]:




