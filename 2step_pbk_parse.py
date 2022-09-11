#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[2]:


get_ipython().run_line_magic('watermark', '')


# In[ ]:





# In[3]:


import time
notebookstart = time.time()


# In[4]:


import os

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
#from webdriver_manager.firefox import GeckoDriverManager

from scipy import stats as sts
import requests as rq
from bs4 import BeautifulSoup as bs
import re

#import time
from typing import Tuple, Optional, List
from tqdm.auto import tqdm
tqdm.pandas()

import pandas as pd
import numpy as np


# In[5]:


# ------ если используется multiprocessing должно быть откомментировано
import multiprocessing as mp

# ------ если используется ray должно быть откомментировано
import ray
ray.init()


# In[ ]:





# Переменные

# In[6]:


DIR_DATA = os.path.join(os.getcwd(), 'data')


# In[7]:


# минимальная и максимальная задержка при загрузке страниц
MIN_DELAY = 2.673 #2.17 #2.673
MAX_DELAY = 5.386 #4.8 #7.22 #9.181


# In[ ]:





# In[8]:





# # Загрузка данных

# In[9]:


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'))
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test.csv'))

df_train['publish_date'] = pd.to_datetime(df_train['publish_date'])
df_test['publish_date']  = pd.to_datetime(df_test['publish_date'])

df_train.shape, df_test.shape


# In[ ]:





# # Сохранение статей с РБК

# In[ ]:


class UserEmulate:
    def __init__(self, inp_min_delay: float, inp_max_delay: float) -> None:
        self.min_delay = inp_min_delay
        self.max_delay = inp_max_delay
        
        self.last_time = time.time()
        self.numb_load = 0


        
    def reset(self, inp_min_delay: Optional[float], inp_max_delay: Optional[float]) -> None:
        """
        Сброс парметров и выставление новых мин и макс задержки
        args
            inp_min_delay - минимальная задержка между загрузками страниц (опционально)
            inp_max_delay - максимальная задержка между загрузками страниц (опционально)
        """
        self.last_time = time.time()
        self.numb_load = 0
        
        if isinstance(inp_min_delay, float):
            self.min_delay = inp_min_delay
            
        if isinstance(inp_max_delay, float):
            self.max_delay = inp_max_delay
       
    
    
    def updatecurrentstate(self):
        """
        Обновление внутреннего состояния класса
        """
        self.last_time = time.time()
        self.numb_load += 1
        

        
    def pauserealuseremulate(self) -> None:
        """
        Эмуляция задержки между кликами пользователя.
        Каждый седьмой клик из нормального распределения
        Каждый третий (при не кратности 7) из хи-квадрат
        Остальные из гамма
        """
        if self.numb_load %7 == 0:
            pause_time = sts.norm.rvs(loc=2, scale=3, size=1)[0]
        elif self.numb_load %3 == 0:
            pause_time = sts.chi2.rvs(df = 1.7, loc = 0, scale = 1, size=1)[0]
        else:
            pause_time = sts.gamma.rvs(a = 1, loc = 1, scale = 2, size=1)[0]

        if (time.time() - self.last_time) > pause_time:
            self.updatecurrentstate()
            return

        if pause_time >= self.min_delay and pause_time <= self.max_delay:
            #print(pause_time)
            time.sleep(pause_time - abs(time.time() - self.last_time))
            self.updatecurrentstate()
            pass
        else:
            pauserealuseremulate()

        return


# In[31]:


category_decode = {
    '5409f11ce063da9c8b588a12':{'name': 'Политика',      # / rbcfreenews
                                'link': 'politics',      # слово в ссылкена рбк
                                'last_word': 'Политика', # возможное последнее слово в title
                                                         # при наличии ошибки
                                },
    '5433e5decbb20f277b20eca9':{'name': 'Общество',      # / photoreport
                                'link': 'society',       # слово в ссылкена рбк
                                'last_word': 'Общество', # возможное последнее слово в title
                                                         # при наличии ошибки
                                },
    '540d5eafcbb20f2524fc0509':{'name': 'Бизнес',        # / rbcfreenews
                                'link': 'business',      # слово в ссылкена рбк
                                'last_word': 'Бизнес',   # возможное последнее слово в title
                                                         # при наличии ошибки
                                },
    '5409f11ce063da9c8b588a13':{'name': 'Экономка',      # / rbcfreenews
                                'link': 'economics',     # слово в ссылкена рбк
                                'last_word': 'Экономика', # возможное последнее слово в title
                                                         # при наличии ошибки
                                },
    '540d5ecacbb20f2524fc050a':{'name': 'Технологии и медия',# / rbcfreenews
                                'link': 'technology_and_media',      # слово в ссылкена рбк
                                'last_word': 'медиа',    # возможное последнее слово в title
                                                         # при наличии ошибки
                                },
    '5409f11ce063da9c8b588a18':{'name': 'Финансы',       # / rbcfreenews
                                'link': 'finances',      # слово вссылкена рбк
                                'last_word': 'Финансы',  # возможное последнее слово в title
                                                         # при наличии ошибки
                                },   

## DELETED????
        '5e54e2089a7947f63a801742':{'name': 'Политика',  # / rbcfreenews
                                'link': 'politics',      # слово вссылкена рбк
                                'last_word': 'Политика', # возможное последнее слово в title
                                                         # при наличии ошибки
                                },  
        '552e430f9a79475dd957f8b3':{'name': 'Деньги',    # / rbcfreenews
                                'link': 'money',         # слово вссылкена рбк
                                'last_word': 'Деньги',   # возможное последнее слово в title
                                                         # при наличии ошибки
                                },  
        '5e54e22a9a7947f560081ea2':{'name': 'Недвижимость',# / city
                                'link': 'realty',          # слово вссылкена рбк
                                'last_word': 'Недвижимость',  # возможное последнее слово в title
                                                           # при наличии ошибки
                                },  
}


# In[32]:


# очистка текста
clean_text = lambda x:' '.join(re.sub('\n|\r|\t|[^а-яА-Яa-zA-Z]', ' ', x).split()) #.lower()


# In[33]:


def get_article_data(inp_df: pd.DataFrame) -> bool:
    """Загрузка и сохранение страницы при помощи selenium c 
    минимальной эмуляцией поведения человека
    
    args
        inp_df - строка для которой необходимо загрузить страницу
                 на основе document_id
        
    return
        True  - страница загружена и сохранена
        False - проблеммы загрузки страницы (404). страница не сохранена
    """
    
    ue = UserEmulate(MIN_DELAY, MAX_DELAY)
    
    date = inp_df[1].strftime('%d/%m/%Y')
    category = category_decode[inp_df[2]]['link']
    link_hash = inp_df[0][:24]
    #print(f'https://www.rbc.ru/{category}/{date}/{link_hash}')
    url = f'https://www.rbc.ru/{category}/{date}/{link_hash}'
    driver.get(url)
    
    # эмулируем задержки пользователя
    ue.pauserealuseremulate()
    
    # 404
    if len(driver.find_elements_by_class_name('error__title')) != 0:
        # общество может быть объеденено с городом
        # а политика часто с новостями
        if inp_df[2] != '5433e5decbb20f277b20eca9':
            category = 'rbcfreenews'
            print(f'https://www.rbc.ru/{category}/{link_hash}')
            url = f'https://www.rbc.ru/{category}/{link_hash}'
        else:
            category = 'city'
            print(f'https://www.rbc.ru/{category}/{date}/{link_hash}')
            url = f'https://www.rbc.ru/{category}/{date}/{link_hash}'
            
        driver.get(url)
        
        # эмулируем задержки пользователя
        ue.pauserealuseremulate()
    
        # документ вне категории и rbcfreenews
        if len(driver.find_elements_by_class_name('error__title')) != 0:
            print(inp_df)
            return False
            
    with open(os.path.join(DIR_DATA, 'pages', f'{inp_df[0]}.html'), 'w',  encoding="utf-8") as f:
        f.write(driver.page_source)
    
    return True


# In[34]:


#  в финальном варианте не используется
def check_for_news(inp_df: pd.DataFrame) -> str:
    """Проверка есть ли статья по адресу в соответствии с категорией
    или только в разделе новости с возвращением определенной категории
    
    args
        inp_df - строка для которой необходимо загрузить страницу
                 на основе document_id
                 
    return
        str - установленная категория
    """
    
    
    ue = UserEmulate(MIN_DELAY, MAX_DELAY)
    
    date = inp_df[1].strftime('%d/%m/%Y')
    category = category_decode[inp_df[2]]['link']
    link_hash = inp_df[0][:24]
    #print(f'https://www.rbc.ru/{category}/{date}/{link_hash}')
    url = f'https://www.rbc.ru/{category}/{date}/{link_hash}'
    driver.get(url)
    
    # эмулируем задержки пользователя
    ue.pauserealuseremulate()

    # 404
    if len(driver.find_elements_by_class_name('error__title')) != 0:
        # общество может быть объеденено с городом
        # а политика часто с новостями
        if inp_df[2] != '5433e5decbb20f277b20eca9':
            category = 'rbcfreenews'
            print(f'https://www.rbc.ru/{category}/{link_hash}')
            url = f'https://www.rbc.ru/{category}/{link_hash}'
        else:
            category = 'city'
            print(f'https://www.rbc.ru/{category}/{date}/{link_hash}')
            url = f'https://www.rbc.ru/{category}/{date}/{link_hash}'
            
        driver.get(url)
        
        # эмулируем задержки пользователя
        ue.pauserealuseremulate()
    
        # документ вне категории и rbcfreenews
        if len(driver.find_elements_by_class_name('error__title')) != 0:
            print(inp_df)
            return 'unknown'
        else:
            return category
            
    
    return category


# для сохранения статей следует откомментировать и выполнить 2 нижележащих блока
# на ubuntu 22 есть проблеммы с Firefox и wewbdriver, так что добавлен chrome
#driver = webdriver.Firefox(executable_path = "C:\\WebDrivers\\bin\\geckodriver")
#driver = webdriver.Firefox(executable_path = '/usr/local/bin/geckodriver')
#driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())

driver = webdriver.Chrome(ChromeDriverManager().install())

tmp = df_train.loc[:, ['document_id', 'publish_date', 'category']].progress_apply(get_article_data, axis = 1)
#tmp = df_train.loc[5000:, ['document_id', 'publish_date', 'category']].progress_apply(get_article_data, axis = 1)# на ubuntu 22 есть проблеммы с Firefox и wewbdriver, так что добавлен chrome
#driver = webdriver.Firefox(executable_path = "C:\\WebDrivers\\bin\\geckodriver")
#driver = webdriver.Firefox(executable_path = '/usr/local/bin/geckodriver')
#driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())

driver = webdriver.Chrome(ChromeDriverManager().install())
tmp = df_test.loc[:, ['document_id', 'publish_date', 'category']].progress_apply(get_article_data, axis = 1)
#tmp = df_test.loc[2000:, ['document_id', 'publish_date', 'category']].progress_apply(get_article_data, axis = 1)
print(sum(tmp))
# проверка на категорию (не участвует в финальном решении)
# на ubuntu 22 есть проблеммы с Firefox и wewbdriver, так что добавлен chrome
#driver = webdriver.Firefox(executable_path = "C:\\WebDrivers\\bin\\geckodriver")
#driver = webdriver.Firefox(executable_path = '/usr/local/bin/geckodriver')
#driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())

driver = webdriver.Chrome(ChromeDriverManager().install())
df_train['link_part'] = df_train.loc[:, ['document_id', 'publish_date', 'category']].progress_apply(check_for_news, axis = 1)

df_train.to_csv(os.path.join(DIR_DATA, 'train_link.csv'), index = False)
# In[ ]:





# In[ ]:





# # Извлекаем признаки из сохраненный статей

# In[12]:


def get_article_info(inp_id: str) -> Tuple[str, str, int, str, int, int, str]:
    """Извлечение признаков из сохраненных статей
    
    args
        inp_id - id статьи (имя под которым она сохранена)
    return
        tuple
           str - категория статьи, полученная со страницы статьи
           str - заголовок статьи (без лишних символов как в оригинальном датасете)
           int - количество картинок в статье
           str - текст обзора статьи
           int - длина текста в словах
           int - наличие 2х статей на одной странице (скрол)
           str - текст статьи
    """
    
    with open(os.path.join(DIR_DATA, 'pages', f'{inp_id}.html'), 'r', encoding="utf-8") as page:
        page_data = page.read()
    
    soup = bs(page_data, 'html.parser')

    # title_info
    tmp_group = soup.find_all('div', attrs={'class': 'article__header__info-block'})
    #if len(tmp_group) != 1:
    if len(tmp_group) == 0:
        print(f'{inp_id} something went wrong. header info block {len(tmp_group)}')
    else:
        if len(tmp_group) > 1:
            two_articles = 1
        else:
            two_articles = 0
        tmp_group = tmp_group[0]


    true_category = tmp_group.find_all('a')
    if len(true_category) != 1:
        print(f'{inp_id} something went wrong. true_category {len(true_category)}')
    true_category = true_category[0].text


    # article header
    tmp_group = soup.find_all('div', attrs={'class': 'article__header__title'})
    if len(tmp_group) == 0:
        print(f'{inp_id} something went wrong. article header {len(tmp_group)}')
    else:
        tmp_group = tmp_group[0]


    true_title = tmp_group.find_all('h1', attrs = {'class': 'article__header__title-in js-slide-title'})
    if len(true_title) != 1:
        true_title = tmp_group.find_all('h1', attrs = {'class': 'article__header__title-in js-slide-title article__header__title-in_relative'})
        if len(true_title) != 1:
            print(f'{inp_id} something went wrong. true_title {len(true_title)}')            
    true_title = text = clean_text(true_title[0].text)


    # article text
    tmp_group = soup.find_all('div', attrs={'class': 'article__text article__text_free'})
    if len(tmp_group) == 0:
        print(f'{inp_id} something went wrong. article text {len(tmp_group)}')
    else:
        tmp_group = tmp_group[0]

    overview = tmp_group.find_all('div', attrs={'class': 'article__text__overview'})
    if len(overview) == 0:
        # rbcfreenews can be without overview
        overview = ''
    else:
        overview = overview[0].span.text

    text = tmp_group.find_all('p')
    if len(text) == 0:
        print(f'{inp_id} something went wrong. text')
    text = ' '.join([clean_text(el.text) for el in text])    


    # images
    tmp_imgs = soup.find_all('div', attrs={'class': 'gallery_vertical__item'})
    #f len(tmp_imgs) != 1:
    #   print(f'something went wrong. article images {len(tmp_imgs)}')
    tmp_imgs = len(tmp_imgs)

    return (true_category, true_title, tmp_imgs, overview, len(text.split()), two_articles, text) #snd_header, snd_text


# In[19]:


#------ если используется ray, то декоратор должен быть откомменирован
#------ при использовании multiprocessing декоратор должен быть закомменитирован
@ray.remote
def parallelize_get_article_info(inp_df: pd.DataFrame) -> pd.DataFrame:
    """Подфункция для !параллельного! вычисления признаков на основании страницы статьи
    
    args
        inp_df - часть основного датасета для которого небходимо вычислить
                 признаки на основаниистраницы статьи
        
    return
        DataFrame, дополненный признаком на основании страницы статьи
    """
    inp_df['tmp'] = inp_df.document_id.apply(get_article_info)
    
    return inp_df


# In[14]:


def make_article_features_mp(inp_df: pd.DataFrame, use_cpu: int) -> pd.DataFrame:
    """Функция для вычисления признаков на основании страницы статьи
    
    args
        inp_df  - DataFrame с document_id? который необходимо дополнить признаками 
                  на основании страницы статьи
        use_cpu - количество процессоров для использвания при параллельном вычислении
        
    return
        DataFrame, дополненный признаками на основании страницы статьи
    """
    # последовательно разбиваем dataframe на части
    split_dfs = np.array_split(inp_df, use_cpu)

    
    
    # вычисляем параллельно
    
    # ------ начало блока, если используется multiprocessing
    #mppool = mp.Pool(processes = use_cpu)
    #pool_results = mppool.map(parallelize_get_article_info, split_dfs)
    #ppool.close()
    #ppool.join()
    # ------ конец блока, если используется multiprocessing

    
    
    # ------ начало блока, если используется ray
    pool_results = [0]*use_cpu
    for el in range(use_cpu):
        pool_results[el] = parallelize_get_article_info.remote(split_dfs[el])
    pool_results = ray.get(pool_results)
    # ------ конец блока, если используется ray
    
    # соединяем части результата
    parts = pd.concat(pool_results, axis=0)

    # выделяем признаки из tuple (можно через to_list)
    parts['true_category'] = parts.tmp.apply(lambda x: x[0])
    parts['true_title'] = parts.tmp.apply(lambda x: x[1])
    parts['nimgs'] = parts.tmp.apply(lambda x: x[2])
    parts['overview'] = parts.tmp.apply(lambda x: x[3])
    parts['text_len'] = parts.tmp.apply(lambda x: x[4])
    parts['two_articles'] = parts.tmp.apply(lambda x: x[5])
    
    parts.drop(['tmp'], axis = 1, inplace = True)
    
    #print(parts.shape)
    print('done')
    
    return parts


# In[ ]:





# In[16]:


get_ipython().run_cell_magic('time', '', "ncpu = os.cpu_count() - 1\nprint('ncpu ', ncpu)\n\nprint('before ', df_train.shape, df_test.shape)\ndf_train = make_article_features_mp(df_train, ncpu)\ndf_test  = make_article_features_mp(df_test, ncpu)\nprint('after  ', df_train.shape, df_test.shape)")

время вычисления. mp & ray примерно равны. секунды варьируются 
train 08:44
train_mp 2:21
train_ray 2:16

test  04:11
test_mp 1:13
test_ray 1:05
# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


df_train.to_csv(os.path.join(DIR_DATA, 'train_extended.csv'), index = False)
df_test.to_csv(os.path.join(DIR_DATA, 'test_extended.csv'), index = False)


# In[ ]:





# In[ ]:


ray.shutdown()


# In[ ]:


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


# In[ ]:




