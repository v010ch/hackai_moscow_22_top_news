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
from itertools import product

import category_encoders as ce
from sklearn import preprocessing
#from sklearn.model_selection import train_test_split

from ast import literal_eval

from tqdm import tqdm
tqdm.pandas()


# In[5]:


get_ipython().run_line_magic('watermark', '--iversions')


# ## Reproducibility block

# In[6]:


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





# In[7]:


DIR_DATA  = os.path.join(os.getcwd(), 'data')


# In[8]:


# ctr для специальных статей по украине
CTR_UKR = 6.096


# энкодеры для кодирования категориальных переменных. 

# но, например, для catboost не требуется такого кодирования, так что оригинальный признак так же останется в датасете,   
# а в модель будут передоваться признаки только через параметр features.

# In[ ]:





# In[9]:


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train_extended.csv'))#, index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test_extended.csv'))#, index_col= 0)

df_train['publish_date'] = pd.to_datetime(df_train['publish_date'])
df_test['publish_date']  = pd.to_datetime(df_test['publish_date'])


# In[10]:


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

# In[11]:


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
        'tags':{'num':  [],   
                'cat':  [],
                'both': [],
                },
         'poly':{'num':  [],   
                'cat':  [],
                'both': [],
                },
        }


# In[12]:


print(df_train.columns.values)


# In[ ]:





# ## Очистка датасета

# этих категорий нет в тесте, а в трейне на них приходится всего 3 записи. они явно лишние.
# 
# уберем статьи раньше минимальной даты в тесте. для начала так, дальше можно будет поиграться.

# In[13]:


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
    
    return inp_df


# In[14]:


min_test_time = pd.Timestamp('2022-01-01')

df_train = clear_data(df_train, min_test_time)


# In[ ]:





# In[ ]:





# ## title

# In[15]:


def add_title_features(inp_df):
    
    # Прямая трансляция, Фоторепортаж, Фотогалерея, Видео, телеканале РБК, Инфографика endswith
    
    inp_df['ph_report']  = inp_df.true_title.apply(lambda x: 1 if x.endswith('Фоторепортаж') else 0)
    inp_df['ph_gallery'] = inp_df.true_title.apply(lambda x: 1 if x.endswith('Фотогалерея') else 0)
    inp_df['tv_prog'] = inp_df.true_title.apply(lambda x: 1 if x.endswith('телеканале РБК') else 0)
    inp_df['online'] = inp_df.true_title.apply(lambda x: 1 if x.endswith('Прямая трансляция') else 0)
    inp_df['video']  = inp_df.true_title.apply(lambda x: 1 if x.endswith('Видео') else 0)
    inp_df['infogr'] = inp_df.true_title.apply(lambda x: 1 if x.endswith('Инфографика') else 0)
    
    inp_df.overview.fillna('', inplace = True)
    inp_df['interview'] = inp_df.overview.apply(lambda x: 1 if 'интервью РБК' in x else 0)
    
    
    if 'video' not in clmns['title']['both']:
        clmns['title']['both'].extend(['ph_report', 'ph_gallery', 'tv_prog', 'online', 'video', 'infogr', 'interview'])
        
    return inp_df


# In[16]:


df_train = add_title_features(df_train)
df_test = add_title_features(df_test)


# In[17]:


df_train.ph_report.sum(), df_train.ph_gallery.sum(), df_train.tv_prog.sum(), df_train.online.sum(), df_train.video.sum(), df_train.infogr.sum()


# In[18]:


df_test.ph_report.sum(), df_test.ph_gallery.sum(), df_test.tv_prog.sum(), df_test.online.sum(), df_test.video.sum(), df_test.infogr.sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # publish date

# In[19]:


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


# In[20]:


border = pd.Timestamp('2022-04-08').date()


# In[21]:


def publish_date_features(inp_df: pd.DataFrame) -> pd.DataFrame:
    
    inp_df['m_d'] = inp_df['publish_date'].dt.date

    inp_df['hour'] = inp_df['publish_date'].dt.hour
    
    inp_df['hour_peak'] = inp_df.hour.apply(lambda x: 1 if x in [4, 12, 16, 21] else 0)
    
    inp_df['dow']  = inp_df['publish_date'].dt.dayofweek
    inp_df['day']    = pd.to_datetime(inp_df['publish_date']).dt.strftime("%d").astype(int)
    inp_df['mounth'] = pd.to_datetime(inp_df['publish_date']).dt.strftime("%m").astype(int)
    
    
    inp_df['holiday'] = inp_df.m_d.apply(lambda x: 1 if x in holidays else 0)
    inp_df['day_before_holiday'] = inp_df.m_d.apply(lambda x: 1 if x in day_before_holiday else 0)
    inp_df['day_after_holiday'] = inp_df.m_d.apply(lambda x: 1 if x in day_after_holiday else 0)
    
    inp_df['distrib_brdr'] = inp_df.m_d.apply(lambda x: 1 if x < border else 0)
    
    
    if 'hour' not in clmns['publish_date']['both']:
        clmns['publish_date']['both'].extend(['hour', 'dow', 'day', 'mounth', 'hour_peak'])#, 'distrib_brdr'])
        
    if 'holiday' not in clmns['publish_date']['both']:
        clmns['publish_date']['both'].extend(['holiday', 'day_before_holiday', 'day_after_holiday', 'distrib_brdr']) 
    
    return inp_df


# In[22]:


print('before ', df_train.shape, df_test.shape)
df_train = publish_date_features(df_train)
df_test  = publish_date_features(df_test)
print('after  ', df_train.shape, df_test.shape)


# In[23]:


print(sum(df_train.holiday), sum(df_train.day_before_holiday), sum(df_train.day_after_holiday), )
print(sum(df_test.holiday), sum(df_test.day_before_holiday), sum(df_test.day_after_holiday), )


# In[ ]:





# In[24]:


hour_cols = ['hour',
             'hour_views_min', 'hour_views_max', 'hour_views_mean', 'hour_views_std',
             'hour_depth_min', 'hour_depth_max', 'hour_depth_mean', 'hour_depth_std',
             'hour_frp_min', 'hour_frp_max', 'hour_frp_mean', 'hour_frp_std',
            ]
hour_stats_start = df_train[df_train.distrib_brdr == 1].groupby(['hour'])[['views', 'depth', 'full_reads_percent']].agg(['min', 'max', 'mean', 'std'])
hour_stats_start = hour_stats_start.reset_index()
hour_stats_start.columns = hour_cols


hour_stats_end = df_train[df_train.distrib_brdr == 0].groupby(['hour'])[['views', 'depth', 'full_reads_percent']].agg(['min', 'max', 'mean', 'std'])
hour_stats_end = hour_stats_end.reset_index()
hour_stats_end.columns = hour_cols



mounth_cols = ['mounth',
             'mounth_views_min', 'mounth_views_max', 'mounth_views_mean', 'mounth_views_std',
             'mounth_depth_min', 'mounth_depth_max', 'mounth_depth_mean', 'mounth_depth_std',
             'mounth_frp_min', 'mounth_frp_max', 'mounth_frp_mean', 'mounth_frp_std',
            ]
mounth_stats_start = df_train[df_train.distrib_brdr == 1].groupby(['mounth'])[['views', 'depth', 'full_reads_percent']].agg(['min', 'max', 'mean', 'std'])
mounth_stats_start = mounth_stats_start.reset_index()
mounth_stats_start.columns = mounth_cols

mounth_stats_end = df_train[df_train.distrib_brdr == 0].groupby(['mounth'])[['views', 'depth', 'full_reads_percent']].agg(['min', 'max', 'mean', 'std'])
mounth_stats_end = mounth_stats_end.reset_index()
mounth_stats_end.columns = mounth_cols



dow_cols = ['dow',
             'dow_views_min', 'dow_views_max', 'dow_views_mean', 'dow_views_std',
             'dow_depth_min', 'dow_depth_max', 'dow_depth_mean', 'dow_depth_std',
             'dow_frp_min', 'dow_frp_max', 'dow_frp_mean', 'dow_frp_std',
            ]
dow_stats_start = df_train[df_train.distrib_brdr == 1].groupby(['dow'])[['views', 'depth', 'full_reads_percent']].agg(['min', 'max', 'mean', 'std'])
dow_stats_start = dow_stats_start.reset_index()
dow_stats_start.columns = dow_cols

dow_stats_end = df_train[df_train.distrib_brdr == 0].groupby(['dow'])[['views', 'depth', 'full_reads_percent']].agg(['min', 'max', 'mean', 'std'])
dow_stats_end = dow_stats_end.reset_index()
dow_stats_end.columns = dow_cols



holiday_cols = ['holiday',
             'holiday_views_min', 'holiday_views_max', 'holiday_views_mean', 'holiday_views_std',
             'holiday_depth_min', 'holiday_depth_max', 'holiday_depth_mean', 'holiday_depth_std',
             'holiday_frp_min', 'holiday_frp_max', 'holiday_frp_mean', 'holiday_frp_std',
            ]
holiday_stats_start = df_train[df_train.distrib_brdr == 1].groupby(['holiday'])[['views', 'depth', 'full_reads_percent']].agg(['min', 'max', 'mean', 'std'])
holiday_stats_start = holiday_stats_start.reset_index()
holiday_stats_start.columns = holiday_cols

holiday_stats_end = df_train[df_train.distrib_brdr == 0].groupby(['holiday'])[['views', 'depth', 'full_reads_percent']].agg(['min', 'max', 'mean', 'std'])
holiday_stats_end = holiday_stats_end.reset_index()
holiday_stats_end.columns = holiday_cols



day_before_holiday_cols = ['day_before_holiday',
             'day_before_holiday_views_min', 'day_before_holiday_views_max', 'day_before_holiday_views_mean', 'day_before_holiday_views_std',
             'day_before_holiday_depth_min', 'day_before_holiday_depth_max', 'day_before_holiday_depth_mean', 'day_before_holiday_depth_std',
             'day_before_holiday_frp_min', 'day_before_holiday_frp_max', 'day_before_holiday_frp_mean', 'day_before_holiday_frp_std',
            ]
day_before_holiday_stats_start = df_train[df_train.distrib_brdr == 1].groupby(['day_before_holiday'])[['views', 'depth', 'full_reads_percent']].agg(['min', 'max', 'mean', 'std'])
day_before_holiday_stats_start = day_before_holiday_stats_start.reset_index()
day_before_holiday_stats_start.columns = day_before_holiday_cols

day_before_holiday_stats_end = df_train[df_train.distrib_brdr == 0].groupby(['day_before_holiday'])[['views', 'depth', 'full_reads_percent']].agg(['min', 'max', 'mean', 'std'])
day_before_holiday_stats_end = day_before_holiday_stats_end.reset_index()
day_before_holiday_stats_end.columns = day_before_holiday_cols




day_after_holiday_cols = ['day_after_holiday',
             'day_after_holiday_views_min', 'day_after_holiday_views_max', 'day_after_holiday_views_mean', 'day_after_holiday_views_std',
             'day_after_holiday_depth_min', 'day_after_holiday_depth_max', 'day_after_holiday_depth_mean', 'day_after_holiday_depth_std',
             'day_after_holiday_frp_min', 'day_after_holiday_frp_max', 'day_after_holiday_frp_mean', 'day_after_holiday_frp_std',
            ]
day_after_holiday_stats_start = df_train[df_train.distrib_brdr == 1].groupby(['day_after_holiday'])[['views', 'depth', 'full_reads_percent']].agg(['min', 'max', 'mean', 'std'])
day_after_holiday_stats_start = day_after_holiday_stats_start.reset_index()
day_after_holiday_stats_start.columns = day_after_holiday_cols

day_after_holiday_stats_end = df_train[df_train.distrib_brdr == 0].groupby(['day_after_holiday'])[['views', 'depth', 'full_reads_percent']].agg(['min', 'max', 'mean', 'std'])
day_after_holiday_stats_end = day_after_holiday_stats_end.reset_index()
day_after_holiday_stats_end.columns = day_after_holiday_cols


# In[ ]:





# In[25]:


def add_daily_stats_date(inp_df:pd.DataFrame, inp_feature, inp_stats_start, inp_stats_end) -> pd.DataFrame:
    
    
    col_x = [el + '_x' for el in inp_stats_start.columns[1:]]
    col_y = [el + '_y' for el in inp_stats_start.columns[1:]]
    
    
    tmp = inp_df[['document_id', inp_feature]].merge(inp_stats_start, on = [inp_feature], how = 'left', validate = 'many_to_one')
    tmp = tmp.merge(inp_stats_end,   on = [inp_feature], how = 'left', validate = 'many_to_one')
    
    
    for el in inp_stats_start.columns[1:]:
        tmp[el] = tmp[f'{el}_x'].fillna(tmp[f'{el}_y'])   

    tmp.drop(col_x, inplace = True, axis = 1)
    tmp.drop(col_y, inplace = True, axis = 1)
    tmp.drop([inp_feature], inplace = True, axis = 1)
    
    
    ret_df = inp_df.merge(tmp, on = ['document_id'], how = 'left', validate = 'one_to_one')
    
    if inp_stats_start.columns[3] not in clmns['publish_date']['num']:
        clmns['publish_date']['num'].extend(inp_stats_start.columns[1:])
    
    return ret_df


# In[ ]:





# In[26]:


print('before ', df_train.shape, df_test.shape, 'add ', hour_stats_start.shape[1])
df_train = add_daily_stats_date(df_train, 'hour', hour_stats_start, hour_stats_end)
df_test  = add_daily_stats_date(df_test, 'hour', hour_stats_start, hour_stats_end)
print('before ', df_train.shape, df_test.shape, 'add ', hour_stats_start.shape[1])


# In[27]:


print('before ', df_train.shape, df_test.shape, 'add ', mounth_stats_start.shape[1])
df_train = add_daily_stats_date(df_train, 'mounth', mounth_stats_start, mounth_stats_end)
df_test  = add_daily_stats_date(df_test, 'mounth', mounth_stats_start, mounth_stats_end)
print('before ', df_train.shape, df_test.shape, 'add ', mounth_stats_start.shape[1])


# In[28]:


print('before ', df_train.shape, df_test.shape, 'add ', dow_stats_start.shape[1])
df_train = add_daily_stats_date(df_train, 'dow', dow_stats_start, dow_stats_end)
df_test  = add_daily_stats_date(df_test, 'dow', dow_stats_start, dow_stats_end)
print('before ', df_train.shape, df_test.shape, 'add ', dow_stats_start.shape[1])


# In[29]:


print('before ', df_train.shape, df_test.shape, 'add ', holiday_stats_start.shape[1])
df_train = add_daily_stats_date(df_train, 'holiday', holiday_stats_start, holiday_stats_end)
df_test  = add_daily_stats_date(df_test, 'holiday', holiday_stats_start, holiday_stats_end)
print('before ', df_train.shape, df_test.shape, 'add ', holiday_stats_start.shape[1])


# In[30]:


print('before ', df_train.shape, df_test.shape, 'add ', day_before_holiday_stats_start.shape[1])
df_train = add_daily_stats_date(df_train, 'day_before_holiday', day_before_holiday_stats_start, day_before_holiday_stats_end)
df_test  = add_daily_stats_date(df_test, 'day_before_holiday', day_before_holiday_stats_start, day_before_holiday_stats_end)
print('before ', df_train.shape, df_test.shape, 'add ', day_before_holiday_stats_start.shape[1])


# In[31]:


print('before ', df_train.shape, df_test.shape, 'add ', day_after_holiday_stats_start.shape[1])
df_train = add_daily_stats_date(df_train, 'day_after_holiday', day_after_holiday_stats_start, day_after_holiday_stats_end)
df_test  = add_daily_stats_date(df_test, 'day_after_holiday', day_after_holiday_stats_start, day_after_holiday_stats_end)
print('before ', df_train.shape, df_test.shape, 'add ', day_after_holiday_stats_start.shape[1])


# In[ ]:





# Рассчитаем дневные статистики + лаги за 7 дней + разница за 7 дней + гаусиана-тренд

# In[32]:


df_train.sort_values(by='m_d').m_d.diff().value_counts()


# In[33]:


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
    
    
    # не учитывать результат сегодняшнего дня
    for col in new_cols:
        ret_df[col] = ret_df[col].shift(1)
        
    v_std = np.std(ret_df.views_mean)
    d_std = np.std(ret_df.depth_mean)
    f_std = np.std(ret_df.frp_mean)
        
    ret_df['view_gaus_2'] = ret_df.views_mean.rolling(2, win_type='gaussian').sum(std = v_std)
    ret_df['depth_gaus_2'] = ret_df.depth_mean.rolling(2, win_type='gaussian').sum(std = d_std)
    ret_df['frp_gaus_2'] = ret_df.frp_mean.rolling(2, win_type='gaussian').sum(std = f_std)
    
    ret_df['view_gaus_3'] = ret_df.views_mean.rolling(3, win_type='gaussian').sum(std = v_std)
    ret_df['depth_gaus_3'] = ret_df.depth_mean.rolling(3, win_type='gaussian').sum(std = d_std)
    ret_df['frp_gaus_3'] = ret_df.frp_mean.rolling(3, win_type='gaussian').sum(std = f_std)
    
    ret_df['view_gaus_7'] = ret_df.views_mean.rolling(7, win_type='gaussian').sum(std = v_std)
    ret_df['depth_gaus_7'] = ret_df.depth_mean.rolling(7, win_type='gaussian').sum(std = d_std)
    ret_df['frp_gaus_7'] = ret_df.frp_mean.rolling(7, win_type='gaussian').sum(std = f_std)
    

    for col, lag in  product(new_cols, list(range(max_lags))):
        ret_df[f'{col}_lag{lag+1}'] = ret_df[col].shift(lag+1)
        ret_df[f'{col}_dif{lag+1}'] = ret_df[col].diff(lag+1)
        #????fillna
        #ret_df[f'{col}_lag{lag+1}'].fillna('mean', inplace = True)
    

    
    return ret_df

daily_stats = create_daily_stats(df_train)
daily_stats.to_csv(os.path.join(DIR_DATA, 'dayly_stats.csv'), index = False)
# In[34]:


daily_stats_start = create_daily_stats(df_train[df_train.distrib_brdr == 1])
daily_stats_start.to_csv(os.path.join(DIR_DATA, 'daily_stats_start.csv'), index = False)

daily_stats_end = create_daily_stats(df_train[df_train.distrib_brdr == 0])
daily_stats_end.to_csv(os.path.join(DIR_DATA, 'daily_stats_end.csv'), index = False)


daily_stats_start.fillna(daily_stats_start.mean(), inplace = True)
daily_stats_end.fillna(daily_stats_end.mean(), inplace = True)

for el in daily_stats_start.columns:
    if sum(daily_stats_start[el].isna()) != 0:
        print(el, sum(daily_stats_start[el].isna()))
        
    if sum(daily_stats_end[el].isna()) != 0:
        print(el, sum(daily_stats_end[el].isna()))


# In[35]:


#daily_stats


# In[ ]:





# In[36]:


daily_stats_start.columns[:]


# Добавим их к датасетам

# In[37]:


def add_daily_stats(inp_df:pd.DataFrame) -> pd.DataFrame:
    
    
    col_x = [el + '_x' for el in daily_stats_start.columns[1:]]
    col_y = [el + '_y' for el in daily_stats_start.columns[1:]]
    
    
    tmp = inp_df[['document_id', 'm_d']].merge(daily_stats_start, on = ['m_d'], how = 'left', validate = 'many_to_one')
    #print(tmp.shape)
    tmp = tmp.merge(daily_stats_end, on = ['m_d'], how = 'left', validate = 'many_to_one')
    #print(tmp.shape)
    
    for el in daily_stats_start.columns[1:]:
        tmp[el] = tmp[f'{el}_x'].fillna(tmp[f'{el}_y'])   

    tmp.drop(col_x, inplace = True, axis = 1)
    tmp.drop(col_y, inplace = True, axis = 1)
    tmp.drop(['m_d'], inplace = True, axis = 1)
    
    
    ret_df = inp_df.merge(tmp, on = ['document_id'], how = 'left', validate = 'one_to_one')
    
    if daily_stats_start.columns[3] not in clmns['publish_date']['num']:
        clmns['publish_date']['num'].extend(daily_stats_start.columns[1:])
    
    return ret_df


# In[38]:


def add_daily_stats___(inp_df:pd.DataFrame) -> pd.DataFrame:
    
    #ret_df = inp_df.merge(daily_stats, on = 'm_d', validate = 'many_to_one')
    ret_df = inp_df.merge(daily_stats, on = 'm_d', how = 'left', validate = 'many_to_one')
    
    if 'views_min' not in clmns['publish_date']['num']:
        clmns['publish_date']['num'].extend(daily_stats.columns[1:])
    
    return ret_df


# In[39]:


print('before ', df_train.shape, df_test.shape, 'add ', daily_stats_start.shape[1])
df_train = add_daily_stats(df_train)
df_test  = add_daily_stats(df_test)
print('after  ', df_train.shape, df_test.shape)


# Проверим на пропуски в тесте

# In[40]:


df_test[['views_min', 'views_max', 'views_mean', 'views_std',
            'depth_min', 'depth_max', 'depth_mean', 'depth_std',
            'frp_min',   'frp_max',   'frp_mean',   'frp_std']].isnull().sum()


# да, на начальныз лагах есть пропуски.    
# заменять будем уже при подборе и построении моделей   

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # title 2
%%time
tmp = df_train[['m_d', 'true_title']]
tmp = pd.concat([tmp, df_test[['m_d', 'true_title']]], 
                ignore_index = True, axis = 0)

tmp['covid']     = tmp.true_title.apply(lambda x: 1 if 'COVID' in x else 0)
tmp['ukr']       = tmp.true_title.apply(lambda x: 1 if 'Укр' in x else 0)
tmp['sanctions'] = tmp.true_title.apply(lambda x: 1 if 'анкц' in x else 0)

#tmp.groupby('m_d')[['covid', 'sanctions']].agg('size')

df_train = df_train.merge(tmp.groupby('m_d')[['covid']].agg('size').rename('covid'),
                          on = ['m_d'], how = 'left', validate = 'many_to_one'
                         )
df_tets = df_test.merge(tmp.groupby('m_d')[['covid']].agg('size').rename('covid'),
                          on = ['m_d'], how = 'left', validate = 'many_to_one'
                         )

df_train = df_train.merge(tmp.groupby('m_d')[['ukr']].agg('size').rename('ukr'),
                          on = ['m_d'], how = 'left', validate = 'many_to_one'
                         )
df_test = df_test.merge(tmp.groupby('m_d')[['ukr']].agg('size').rename('ukr'),
                          on = ['m_d'], how = 'left', validate = 'many_to_one'
                         )


df_train = df_train.merge(tmp.groupby('m_d')[['sanctions']].agg('size').rename('sanctions'),
                          on = ['m_d'], how = 'left', validate = 'many_to_one'
                         )
df_test = df_test.merge(tmp.groupby('m_d')[['sanctions']].agg('size').rename('sanctions'),
                          on = ['m_d'], how = 'left', validate = 'many_to_one'
                         )

clmns['title']['num'].extend(['sanctions', 'ukr', 'covid'])df_train.shape, df_test.shape
# In[ ]:





# In[ ]:





# ## session

# In[ ]:





# ## authors

# Авторы считываются как строки, а не как массив строк. исправим.

# In[41]:


def prep_authors(inp_df): 

    
    inp_df["authors_int"] = inp_df.authors.astype('category')
    inp_df["authors_int"] = inp_df.authors_int.cat.codes
    inp_df["authors_int"] = inp_df.authors_int.astype('int')
    
    
    inp_df['authors'] = inp_df.authors.apply(lambda x: literal_eval(x))
    inp_df['authors'] = inp_df.authors.apply(lambda x: x if len(x) > 0 else ['without_author'])
    
    inp_df['Nauthors']   = inp_df.authors.apply(lambda x: len(x))
    inp_df['Nauthors_2'] = inp_df.Nauthors.apply(lambda x: 1 / (x+1)**2)
    inp_df['Nauthors_3'] = inp_df.Nauthors.apply(lambda x: -1 / (x+1)**2)
    
    if 'authors_int' not in clmns['authors']['num']:
        clmns['authors']['num'].extend(['authors_int'])
    
    if 'Nauthors' not in clmns['authors']['num']:
        clmns['authors']['num'].extend(['Nauthors', 'Nauthors_2', 'Nauthors_3'])
    
    return inp_df

df_train['authors']  = df_train.authors.apply(lambda x: literal_eval(x))
df_test['authors']   = df_test.authors.apply( lambda x: literal_eval(x))

# пустое поле автора заменим на значение, что автор не указан
df_train['authors'] = df_train['authors'].apply(lambda x: x if len(x) > 0 else ['without_author'])
df_test['authors']  = df_test['authors'].apply( lambda x: x if len(x) > 0 else ['without_author'])df_train['Nauthors'] = df_train.authors.apply(lambda x: len(x))
df_test['Nauthors']  = df_test.authors.apply(lambda x: len(x))clmns['authors']['num'].extend(['Nauthors'])
# In[42]:


print('before ', df_train.shape, df_test.shape)
df_train = prep_authors(df_train)
df_test  = prep_authors(df_test)
print('after  ', df_train.shape, df_test.shape)


# In[ ]:





# выделяем всех авторов в трейне

# In[43]:


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

# In[44]:


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

# In[45]:


auth_doc_id = {el: [] for el in all_authors}

for el in tqdm(df_train.loc[:, ['document_id', 'authors']].values):
    for athr in range(len(el[1])):
        auth_doc_id[el[1][athr]].append(el[0])
        
with open(os.path.join(DIR_DATA, 'authors_all.pkl'), 'wb') as pkl_file:
    pkl.dump(auth_doc_id, pkl_file)


# Статьи только автора (в одиночку)(пока не применяется)

# In[46]:


auth_doc_id_alone = {el: [] for el in all_authors}

for el in tqdm(df_train.loc[:, ['document_id', 'authors']].values):
    if len(el[1]) == 1:
        auth_doc_id_alone[el[1][0]].append(el[0])
        
with open(os.path.join(DIR_DATA, 'authors_alone.pkl'), 'wb') as pkl_file:
    pkl.dump(auth_doc_id_alone, pkl_file)


# In[ ]:





# Соберем статистику по авторам (с учетом совместных)

# In[47]:


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


# In[48]:


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

df_train.drop(['cur_author'], inplace = True, axis = 1)


# In[49]:


df_author.to_csv(os.path.join(DIR_DATA, 'author_together.csv'), index = False)


# In[50]:


#df_author.tail()


# In[ ]:





# Добавляем статистики по авторам в датасеты

# In[51]:


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
                #divisor -= 1
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


# In[52]:


def add_author_features(inp_df):
    
    tmp_cols = inp_df.columns
    author_stats = inp_df[['authors']].progress_apply(add_author_statistics, axis = 1)
    inp_df = pd.concat([inp_df, 
                      pd.DataFrame(author_stats.to_list(), columns = author_columns[2:])], 
                      ignore_index = True, axis = 1)
    
    inp_df.columns = list(tmp_cols) + list(author_columns[2:])
    
    if author_columns[5] not in clmns['authors']['num']:
        clmns['authors']['num'].extend(author_columns[2:]) 
    
    return inp_df


# In[53]:


# кроме полей author / author_size
print('before', df_train.shape, df_test.shape)
df_train = add_author_features(df_train)
df_test  = add_author_features(df_test)
print('after', df_train.shape, df_test.shape)


# In[54]:


#clmns['authors']['num'].extend(author_columns[2:])


# In[55]:


#list(tmp_cols) + list(author_columns[2:])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## ctr

# In[56]:


crt_replace = df_train[df_train.ctr > 0].ctr.mean()
crt_replace


# In[57]:


def add_ctr_features(inp_df):
    
    #inp_df['spec_event_1'] = inp_df.ctr.apply(lambda x: 1 if x == 6.096 else 0)
    
    inp_df.ctr.replace(0.0, crt_replace, inplace = True)
    inp_df['ctr_2'] = inp_df.ctr.apply(lambda x: np.sqrt(x))
    
    #if 'spec_event_1' not in clmns['ctr']['both']:
    #    clmns['ctr']['both'].extend(['spec_event_1']) 
    
    if 'ctr_2' not in clmns['ctr']['num']:
        clmns['ctr']['num'].extend(['ctr_2']) 
                                       
    return inp_df


# In[58]:


print('before ', df_train.shape, df_test.shape)
df_train = add_ctr_features(df_train)
df_test  = add_ctr_features(df_test)
print('after  ', df_train.shape, df_test.shape)


# In[59]:


df_test['spec'] = df_test.ctr.apply(lambda x: 1 if x == CTR_UKR else 0)


# In[ ]:





# In[ ]:





# In[60]:


if 'ctr' not in clmns['ctr']['num']:
    clmns['ctr']['num'].extend(['ctr']) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## category

# Собираем статистики по категориям

# In[61]:


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
    
    for col in new_cols:
        ret_df[col] = ret_df[col].shift(1)
    
    
    v_std = np.std(ret_df.cat_views_mean)
    d_std = np.std(ret_df.cat_depth_mean)
    f_std = np.std(ret_df.cat_frp_mean)
    
    ret_df['cat_view_gaus_2'] = ret_df.cat_views_mean.rolling(2, win_type='gaussian').sum(std = v_std)
    ret_df['cat_depth_gaus_2'] = ret_df.cat_depth_mean.rolling(2, win_type='gaussian').sum(std = d_std)
    ret_df['cat_frp_gaus_2'] = ret_df.cat_frp_mean.rolling(2, win_type='gaussian').sum(std = f_std)
    
    ret_df['cat_view_gaus_3'] = ret_df.cat_views_mean.rolling(3, win_type='gaussian').sum(std = v_std)
    ret_df['cat_depth_gaus_3'] = ret_df.cat_depth_mean.rolling(3, win_type='gaussian').sum(std = d_std)
    ret_df['cat_frp_gaus_3'] = ret_df.cat_frp_mean.rolling(3, win_type='gaussian').sum(std = f_std)
    
    ret_df['cat_view_gaus_7'] = ret_df.cat_views_mean.rolling(7, win_type='gaussian').sum(std = v_std)
    ret_df['cat_depth_gaus_7'] = ret_df.cat_depth_mean.rolling(7, win_type='gaussian').sum(std = d_std)
    ret_df['cat_frp_gaus_7'] = ret_df.cat_frp_mean.rolling(7, win_type='gaussian').sum(std = f_std)
    
    for col, lag in  product(new_cols, list(range(max_lags))):
        ret_df[f'{col}_lag{lag+1}'] = ret_df[col].shift(lag+1)
        ret_df[f'{col}_dif{lag+1}'] = ret_df[col].diff(lag+1)
        
        #????fillna
        #ret_df[f'{col}_lag{lag+1}'].fillna('mean', inplace = True)
    

        
    return ret_df

if not os.path.exists(os.path.join(DIR_DATA, 'daily_stats_category.csv')):
    daily_stats_category = create_daily_stats_by_category(df_train)
    daily_stats_category.to_csv(os.path.join(DIR_DATA, 'daily_stats_category.csv'), index = False)
else:
    daily_stats_category = pd.read_csv(os.path.join(DIR_DATA, 'daily_stats_category.csv'))#, index = False)
# In[62]:


daily_stats_cat_start = create_daily_stats_by_category(df_train[df_train.distrib_brdr == 1])
daily_stats_cat_start.to_csv(os.path.join(DIR_DATA, 'daily_stats_cat_start.csv'), index = False)

daily_stats_cat_end = create_daily_stats_by_category(df_train[df_train.distrib_brdr == 0])
daily_stats_cat_end.to_csv(os.path.join(DIR_DATA, 'daily_stats_cat_end.csv'), index = False)


daily_stats_cat_start.fillna(daily_stats_cat_start.mean(), inplace = True)
daily_stats_cat_end.fillna(daily_stats_cat_end.mean(), inplace = True)

for el in daily_stats_cat_start.columns:
    if sum(daily_stats_cat_start[el].isna()) != 0:
        print(el, sum(daily_stats_cat_start[el].isna()))
        
    if sum(daily_stats_cat_end[el].isna()) != 0:
        print(el, sum(daily_stats_cat_end[el].isna()))


# In[63]:


#np.std(daily_stats_category.cat_frp_mean)


# In[ ]:





# In[ ]:





# Добавляем статистики по категориям в датасеты

# In[64]:


#df_train.m_d
#daily_stats_cat_end.columns


# In[65]:


def add_daily_stats_category(inp_df:pd.DataFrame) -> pd.DataFrame:
    
    
    col_x = [el + '_x' for el in daily_stats_cat_start.columns[2:]]
    col_y = [el + '_y' for el in daily_stats_cat_start.columns[2:]]
    
    
    tmp = inp_df[['document_id', 'category', 'm_d']].merge(daily_stats_cat_start, on = ['category', 'm_d'], how = 'left', validate = 'many_to_one')
    tmp = tmp.merge(daily_stats_cat_end,   on = ['category', 'm_d'], how = 'left', validate = 'many_to_one')
    
    
    for el in daily_stats_cat_start.columns[2:]:
        tmp[el] = tmp[f'{el}_x'].fillna(tmp[f'{el}_y'])   

    tmp.drop(col_x, inplace = True, axis = 1)
    tmp.drop(col_y, inplace = True, axis = 1)
    tmp.drop(['category', 'm_d'], inplace = True, axis = 1)
    
    
    ret_df = inp_df.merge(tmp, on = ['document_id'], how = 'left', validate = 'one_to_one')
    
    if daily_stats_cat_start.columns[3] not in clmns['category']['num']:
        clmns['category']['num'].extend(daily_stats_cat_start.columns[2:])
    
    return ret_df


# In[66]:


def add_daily_stats_category__(inp_df:pd.DataFrame) -> pd.DataFrame:
    
    ret_df = inp_df.merge(daily_stats_category, on = ['category', 'm_d'], how = 'left', validate = 'many_to_one')
    
    if daily_stats_category.columns[3] not in clmns['category']['num']:
        clmns['category']['num'].extend(daily_stats_category.columns[2:])
    
    return ret_df


# In[67]:


print('before ', df_train.shape, df_test.shape, 'add ', daily_stats_cat_start.shape, len(daily_stats_cat_start.columns))
#print('before ', df_train.shape, df_test.shape, 'add ', daily_stats_category.shape, len(daily_stats_category.columns))
df_train = add_daily_stats_category(df_train)
df_test = add_daily_stats_category(df_test)
print('after  ', df_train.shape, df_test.shape, )

col_x = [el + '_x' for el in daily_stats_cat_start.columns[2:]]
col_y = [el + '_y' for el in daily_stats_cat_start.columns[2:]]

tmp = df_train[['document_id', 'category', 'm_d']].merge(daily_stats_cat_start, on = ['category', 'm_d'], how = 'left', validate = 'many_to_one')
tmp = tmp.merge(daily_stats_cat_end,   on = ['category', 'm_d'], how = 'left', validate = 'many_to_one')

for el in daily_stats_cat_start.columns[2:]:
    tmp[el] = tmp[f'{el}_x'].fillna(tmp[f'{el}_y'])    
#tmp[daily_stats_cat_start.columns[2:]] = tmp[col_x].fillna(tmp[col_y])

tmp.drop(col_x, inplace = True, axis = 1)
tmp.drop(col_y, inplace = True, axis = 1)

for el in daily_stats_cat_start.columns[2:]:
    if sum(tmp[el].isna()) != 0:
        print('wtf ', el, ' ',sum(tmp[el].isna()))  
# In[ ]:





# In[68]:


#print(' '.join(df_train.columns))


# In[ ]:





# In[ ]:





# In[69]:


#clmns['category']['num'].extend(daily_stats_category.columns[2:])


# Проверяем, что все данные есть в тесте

# In[70]:


#df_test[['cat_views_min', 'cat_views_max', 'cat_views_mean', 'cat_views_std',
#                'cat_depth_min', 'cat_depth_max', 'cat_depth_mean', 'cat_depth_std',
#               'cat_frp_min',   'cat_frp_max',   'cat_frp_mean',   'cat_frp_std',]].isnull().sum()
df_test[daily_stats_cat_start.columns[2:]].isnull().sum()
#df_test[daily_stats_category.columns[2:]].isnull().sum()


# Значения в признаках с лагами могут отсутствовать

# In[ ]:





# In[71]:


def prep_category(inp_df):
    
    inp_df["category_int"] = inp_df.category.astype('category')
    inp_df["category_int"] = inp_df.category_int.cat.codes
    inp_df["category_int"] = inp_df.category_int.astype('int')
    
    if 'category_int' not in clmns['category']['num']:
        clmns['category']['num'].extend(['category_int'])
    
    
    if 'category' not in clmns['category']['cat']:
        clmns['category']['cat'].extend(['category'])
    
    return inp_df


# In[72]:


print('before ', df_train.shape, df_test.shape,)
df_train = prep_category(df_train)
df_test = prep_category(df_test)
print('after  ', df_train.shape, df_test.shape, )


# In[73]:


df_train.columns


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## tags

# In[74]:


#df_train['tags']  = df_train.tags.apply(lambda x: literal_eval(x))
#df_test['tags']   = df_test.tags.apply( lambda x: literal_eval(x))


# In[75]:


def add_tags_features(inp_df):
    
    inp_df["tags_int"] = inp_df.tags.astype('category')
    inp_df["tags_int"] = inp_df.tags_int.cat.codes
    inp_df["tags_int"] = inp_df.tags_int.astype('int')
    
    
    inp_df['tags'] = inp_df.tags.apply(lambda x: literal_eval(x))
    inp_df['ntags'] = inp_df.tags.apply(lambda x: len(x))
    
    if 'tags_int' not in clmns['tags']['num']:
        clmns['tags']['num'].extend(['ntags', 'tags_int'])
        
    return inp_df


# In[76]:


print('before ', df_train.shape, df_test.shape, )
df_train = add_tags_features(df_train)
df_test = add_tags_features(df_test)
print('after  ', df_train.shape, df_test.shape, )


# In[ ]:




all_tags = set()
for el in df_train.tags.values:
    if len (el) == 0:
        continue
    if len(el) == 1:
        all_tags.add(el[0])
        continue
        
    for tag in el:
        all_tags.add(tag)
        if not os.path.exists(os.path.join(DIR_DATA, 'df_tags.csv')):
    tag_columns = ['tag', 'm_d', 'tag_size', 'v_tag_min', 'v_tag_max', 'v_tag_mean', 'v_tag_std', 'd_tag_min',
                  'd_tag_max', 'd_tag_mean', 'd_tag_std', 'f_tag_min', 'f_tag_max', 'f_tag_mean', 'f_tag_std',
                 ]

    tag_group_columns = ['tag_size', 
                            'v_tag_min', 'v_tag_max', 'v_tag_mean', 'v_tag_std',
                            'tag_size2',
                            'd_tag_min', 'd_tag_max', 'd_tag_mean', 'd_tag_std',
                            'tag_size3',
                            'f_tag_min', 'f_tag_max', 'f_tag_mean', 'f_tag_std',
                       ]
    
    df_tags = pd.DataFrame(columns = tag_columns)
    for el in tqdm(all_tags):
        # собираем статистики текущего tag
        df_train['cur_tag'] = df_train.tags.apply(lambda x: el if el in x else 0)

        tmp = df_train.groupby(['cur_tag', 'm_d'])[['views', 'depth', 'full_reads_percent']].agg(['size', 'min', 'max', 'mean', 'std'])
        tmp.columns = tag_group_columns
        tmp.reset_index(inplace = True)
        tmp.drop(['tag_size2', 'tag_size3'], axis = 1, inplace = True)

        tmp = tmp[tmp.cur_tag != 0 ]
        df_tags = pd.concat([df_tags, tmp], ignore_index = True, axis = 0)
        df_tags['tag'] = df_tags.cur_tag
        #df_tags.drop(['cur_tag'], inplace = True, axis = 1)
        
    df_tags.to_csv(os.path.join(DIR_DATA, 'df_tags.csv'), index = False) 
else:
    df_tags =  pd.read_csv(os.path.join(DIR_DATA, 'df_tags.csv'))
# In[ ]:





# In[ ]:





# In[77]:


#df_tags


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # text_len

# In[78]:


def add_text_len_features(inp_df):
    
    
    inp_df["text_len_2"] = inp_df.text_len.apply(lambda x: 1 / (x+1)**2)
    inp_df["text_len_3"] = inp_df.text_len.apply(lambda x: np.sqrt(x))
    
    if 'text_len_2' not in clmns['document_id']['num']:
        clmns['document_id']['num'].extend(['text_len_2', 'text_len_3'])
    
    return inp_df


# In[79]:


print('before ', df_train.shape, df_test.shape, )
df_train = add_text_len_features(df_train)
df_test = add_text_len_features(df_test)
print('after  ', df_train.shape, df_test.shape, )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Предобработка признаков в датасетах

# выделяем числовые признаки для нормализации

# In[80]:


cat_cols = []
num_cols = []

for el in clmns.keys():
    cat_cols.extend(clmns[el]['cat'])
    num_cols.extend(clmns[el]['num'])
    if len(clmns[el]['both']) != 0:
        print(clmns[el]['both'])


# In[81]:


num_cols.extend(['hour', 'mounth', 'dow', ])
cat_cols.extend([ 'ph_report', 'ph_gallery', 'tv_prog', 'online', 'video', 'infogr',
                  'holiday', 'day_before_holiday', 'day_after_holiday', 'distrib_brdr',
                  #'spec_event_1'
                ])


# In[ ]:





# In[ ]:





# # Полипризнаки

# In[82]:


poly_cols = ['Nauthors', 'ctr', 'text_len', 'hour', 'day', 'mounth', 'dow', 'nimgs', 'category_int']
len(poly_cols)


# In[83]:


#poly2 = preprocessing.PolynomialFeatures(degree = 2, include_bias = False)
#oly2.fit(df_train[poly_cols])


# In[84]:


#poly3 = preprocessing.PolynomialFeatures(degree = 3, include_bias = False)
#poly3.fit(df_train[poly_cols])


# In[ ]:





# In[85]:


def addd_poly(inp_df, inp_poly):
    
    inp_cols = inp_df.columns
    
    tmp = inp_poly.transform(inp_df[poly_cols])
    tmp = pd.DataFrame(tmp, columns = inp_poly.get_feature_names(poly_cols))
    
    inp_df = pd.concat([inp_df, 
                        tmp.iloc[:, len(poly_cols):]
                       ], ignore_index = True, axis = 1)
    new_cols = list(inp_cols) + list(inp_poly.get_feature_names(poly_cols)[len(poly_cols):])
    
    inp_df.columns = new_cols
         
    if inp_poly.get_feature_names(poly_cols)[-1] not in clmns['poly']['num']:
        clmns['poly']['num'].extend(inp_poly.get_feature_names(poly_cols)[len(poly_cols):])
        
        
    return inp_df


# In[86]:


#print('before ', df_train.shape, df_test.shape)
#df_train = addd_poly(df_train, poly2) #poly3
#df_test  = addd_poly(df_test,  poly2) #poly3
#print('after  ', df_train.shape, df_test.shape)


# In[ ]:





# In[87]:


#num_cols.extend(clmns['poly']['num'])


# In[ ]:





# In[88]:


df_train.to_csv(os.path.join( DIR_DATA, 'train_upd_no_norm.csv'), index = False)
df_test.to_csv(os.path.join( DIR_DATA,  'test_upd_no_norm.csv'), index = False)


# In[ ]:





# нормализуем
#scaler = preprocessing.MinMaxScaler()   #Transform features by scaling each feature to a given range.
#scaler = preprocessing.Normalizer()     #Normalize samples individually to unit norm.
scaler = preprocessing.StandardScaler()  #Standardize features by removing the mean and scaling to unit variance.

scaler.fit(df_train[num_cols])

df_train[num_cols] = scaler.transform(df_train[num_cols])
df_test[num_cols]  = scaler.transform(df_test[num_cols])
# In[89]:


scaler_start = preprocessing.StandardScaler()  #Standardize features by removing the mean and scaling to unit variance.
scaler_end = preprocessing.StandardScaler()  #Standardize features by removing the mean and scaling to unit variance.

scaler_start.fit(df_train[df_train.distrib_brdr == 1][num_cols])
scaler_end.fit(df_train[df_train.distrib_brdr == 0][num_cols])

df_train.loc[df_train.query('distrib_brdr == 1').index, num_cols] = scaler_start.transform(df_train[df_train.distrib_brdr == 1][num_cols])
df_test.loc[df_test.query('distrib_brdr == 1').index, num_cols]  = scaler_start.transform(df_test[df_test.distrib_brdr == 1][num_cols])

df_train.loc[df_train.query('distrib_brdr == 0').index, num_cols] = scaler_end.transform(df_train[df_train.distrib_brdr == 0][num_cols])
df_test.loc[df_test.query('distrib_brdr == 0').index, num_cols]  = scaler_end.transform(df_test[df_test.distrib_brdr == 0][num_cols])


# In[ ]:





# In[90]:


# определяем CTR_UKR спецстатей по украине после нормализации
#for el in doc_id_ukr:
#    print(df_test[df_test.document_id == el].ctr.values)


# In[ ]:





# Добавляем эмбединги

# In[91]:


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
    
# In[92]:


emb_train = pd.read_csv(os.path.join(DIR_DATA, f'ttl_cln_emb_train_{MODEL_FOLDER}_{MAX_LENGTH}_pca{PCA_COMPONENTS}.csv'))
#emb_train.drop(['document_id', 'title'], axis = 1 , inplace = True)
emb_train.drop(['true_title'], axis = 1 , inplace = True)

df_train = df_train.merge(emb_train, on = 'document_id', validate = 'one_to_one')
df_train.shape, emb_train.shape


# In[93]:


emb_test = pd.read_csv(os.path.join(DIR_DATA, f'ttl_cln_emb_test_{MODEL_FOLDER}_{MAX_LENGTH}_pca{PCA_COMPONENTS}.csv'))
#emb_test.drop(['document_id', 'title'], axis = 1 , inplace = True)
emb_test.drop(['true_title'], axis = 1 , inplace = True)

df_test = df_test.merge(emb_test, on = 'document_id', validate = 'one_to_one')
df_test.shape, emb_test.shape


# In[94]:


num_cols = num_cols + list(emb_train.columns)


# In[95]:


if 'document_id' in num_cols:
    num_cols.remove('document_id')


# In[96]:


clmns['title']['num'].extend(emb_train.columns[1:])


# In[ ]:





# In[ ]:





# ## save

# In[97]:


df_test.shape, df_test.shape


# In[98]:


df_train.to_csv(os.path.join( DIR_DATA, 'train_upd.csv'))
df_test.to_csv(os.path.join( DIR_DATA,  'test_upd.csv'))


# In[99]:


with open(os.path.join(DIR_DATA, 'clmns.pkl'), 'wb') as pickle_file:
    pkl.dump(clmns, pickle_file)


# In[ ]:





# In[100]:


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


# In[ ]:





# In[101]:


#clmns


# In[102]:


#cat_cols


# In[103]:


#print(num_cols)


# In[ ]:





# In[ ]:





# In[ ]:




