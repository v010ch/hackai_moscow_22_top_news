#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

from selenium import webdriver
from scipy import stats as sts
import requests as rq
from bs4 import BeautifulSoup as bs
import re

import time
from tqdm.auto import tqdm
tqdm.pandas()

import pandas as pd
import numpy as np


# In[ ]:





# In[2]:


DIR_DATA = os.path.join(os.getcwd(), 'data')


# In[3]:


MIN_DELAY = 2.673 #2.17 #2.673
MAX_DELAY = 5.386 #4.8 #7.22 #9.181


# In[ ]:





# In[4]:


#LOAD_NUMBER = 0
#LAST_LOAD = time.time()


# In[ ]:





# In[5]:


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'))
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test.csv'))

df_train['publish_date'] = pd.to_datetime(df_train['publish_date'])
df_test['publish_date']  = pd.to_datetime(df_test['publish_date'])

df_train.shape, df_test.shape


# In[ ]:





# In[6]:


def pauserealuseremulate(numb_load, last_time):
    
    if numb_load %7 == 0:
        pause_time = sts.norm.rvs(loc=2, scale=3, size=1)[0]
    elif numb_load %3 == 0:
        pause_time = sts.chi2.rvs(df = 1.7, loc = 0, scale = 1, size=1)[0]
    else:
        pause_time = sts.gamma.rvs(a = 1, loc = 1, scale = 2, size=1)[0]
        
    if (time.time() - last_time) > pause_time:
        return
    
    if pause_time >= MIN_DELAY and pause_time <= MAX_DELAY:
        #print(pause_time)
        time.sleep(pause_time - abs(time.time() - last_time))
        pass
    else:
        pauserealuseremulate(numb_load, last_time)
    
    return


# In[7]:


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

5409f11ce063da9c8b588a12

648
https://www.rbc.ru/politics/22/03/2022/623a42f49a7947092e9f8a6byyO-cdUwQK2SyNATEJw4Hg
https://www.rbc.ru/rbcfreenews/623a42f49a7947092e9f8a6b

1511
624463cd9a79476ed1dfc869E_KUrdstQmmg6BC2cXKeQw
https://www.rbc.ru/politics/30/03/2022/624463cd9a79476ed1dfc869

1808
620d619f9a7947376b27bfa4Inynzi57Rha5kH_bTmf7rg
https://www.rbc.ru/rbcfreenews/620d619f9a7947376b27bfa4
    
    
    
    
5433e5decbb20f277b20eca9

1868
623326679a794756b8f2a9689gvGuGVAQdSks_mvJUPH0g
https://www.rbc.ru/photoreport/26/03/2022/623326679a794756b8f2a968

2270
628f46d79a7947a590484e1fCdVz8DVMScWTjWmc-egG6Q
https://www.rbc.ru/society/26/05/2022/628f46d79a7947a590484e1f    

2320
620099d69a794755aba55d7ay3GfLOkyT0udO4mpPvGtpg
https://www.rbc.ru/society/07/02/2022/620099d69a794755aba55d7a
    
176
6244c29b9a79478f9a339bca0RwqdNMOSfeAefxAr_8Wwg
https://www.rbc.ru/society/31/03/2022/6244c29b9a79478f9a339bca
# In[8]:


clean_text = lambda x:' '.join(re.sub('\n|\r|\t|[^а-яА-Яa-zA-Z]', ' ', x).split()) #.lower()


# In[9]:


def get_article_data(inp_df):
    
    global load_number
    global last_load
    
    #article = ''
    
    #print(inp_df[1])
    #print(inp_df[1].dt.date.day)
    date = inp_df[1].strftime('%d/%m/%Y')
    category = category_decode[inp_df[2]]['link']
    link_hash = inp_df[0][:24]
    #print(f'https://www.rbc.ru/{category}/{date}/{link_hash}')
    url = f'https://www.rbc.ru/{category}/{date}/{link_hash}'
    driver.get(url)
    
    # эмулируем задержки пользователя
    pauserealuseremulate(load_number, last_load)
    last_load = time.time()
    load_number += 1
    
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
        pauserealuseremulate(load_number, last_load)
        last_load = time.time()
        load_number += 1
    
        # документ вне категории и rbcfreenews
        if len(driver.find_elements_by_class_name('error__title')) != 0:
            print(inp_df)
            return False
            
    with open(os.path.join(DIR_DATA, 'pages', f'{inp_df[0]}.html'), 'w',  encoding="utf-8") as f:
        f.write(driver.page_source)
    
    return True


# In[10]:


def check_for_news(inp_df):
    
    global load_number
    global last_load
    
    #article = ''
    
    #print(inp_df[1])
    #print(inp_df[1].dt.date.day)
    date = inp_df[1].strftime('%d/%m/%Y')
    category = category_decode[inp_df[2]]['link']
    link_hash = inp_df[0][:24]
    #print(f'https://www.rbc.ru/{category}/{date}/{link_hash}')
    url = f'https://www.rbc.ru/{category}/{date}/{link_hash}'
    driver.get(url)
    
    # эмулируем задержки пользователя
    pauserealuseremulate(load_number, last_load)
    last_load = time.time()
    load_number += 1
    
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
        pauserealuseremulate(load_number, last_load)
        last_load = time.time()
        load_number += 1
    
        # документ вне категории и rbcfreenews
        if len(driver.find_elements_by_class_name('error__title')) != 0:
            print(inp_df)
            return 'unknown'
        else:
            return category
            
    
    return category

load_number = 0
last_load = time.time()

driver = webdriver.Firefox(executable_path = "C:\\WebDrivers\\bin\\geckodriver")
tmp = df_train.loc[5000:, ['document_id', 'publish_date', 'category']].progress_apply(get_article_data, axis = 1)load_number = 0
last_load = time.time()

driver = webdriver.Firefox(executable_path = "C:\\WebDrivers\\bin\\geckodriver")
tmp = df_test.loc[:, ['document_id', 'publish_date', 'category']].progress_apply(get_article_data, axis = 1)
print(sum(tmp))
# In[ ]:


load_number = 0
last_load = time.time()

driver = webdriver.Firefox(executable_path = "C:\\WebDrivers\\bin\\geckodriver")
df_train['link_part'] = df_train.loc[:, ['document_id', 'publish_date', 'category']].progress_apply(check_for_news, axis = 1)

df_train.to_csv(os.path.join(DIR_DATA, 'train_link.csv'), index = False)


# In[ ]:





# In[ ]:





# In[ ]:




# обычная статья
#tst_page = '624ac09c9a7947db3d80c98e.html'
tst_page = '624ac09c9a7947db3d80c98eIDE7mtH4RBqGn-8MXfGffQ.html'

# статья с многго фото
#623326679a794756b8f2a9689gvGuGVAQdSks_mvJUPH0g
#https://www.rbc.ru/photoreport/26/03/2022/623326679a794756b8f2a968
#tst_page = '623326679a794756b8f2a9689gvGuGVAQdSks_mvJUPH0g.html'pages = [el for el in os.listdir(os.path.join(DIR_DATA, 'pages')) if el.endswith('.html')]
doc_id = [el[:-5] for el in pages]

rbk_data = pd.DataFrame(columns = ['document_id', 'true_title', 'text_overview', 'text', 'true_category'])
rbk_data['document_id'] = doc_id
# In[10]:


#(page_data.text, features="lxml") # features="lxml" чтобы не было warning


# In[11]:


def get_article_info(inp_id):
    
    with open(os.path.join(DIR_DATA, 'pages', f'{inp_id}.html'), 'r', encoding="utf-8") as page:
        page_data = page.read()
    
    soup = bs(page_data, 'html.parser')

    # title_info
    tmp_group = soup.find_all('div', attrs={'class': 'article__header__info-block'})
    #if len(tmp_group) != 1:
    if len(tmp_group) == 0:
        print(f'{inp_id} something went wrong. header info block {len(tmp_group)}')
        #if len(tmp_group) > 1:
        #    for el in range(len(tmp_group)):
        #        print(tmp_group[el])
        #raise
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


    #new_views = tmp_group.find_all('span', attrs=  {'class': 'article__header__counter js-insert-views-count'})
    #if len(new_views) != 1:
    #    print(f'something went wrong. new_views {len(new_views)}')
    #new_views = new_views[0].text

    # article header
    tmp_group = soup.find_all('div', attrs={'class': 'article__header__title'})
    if len(tmp_group) == 0:
        print(f'{inp_id} something went wrong. article header {len(tmp_group)}')
        # raise
    else:
        #if len(tmp_group) > 1:
        #    snd_header
        tmp_group = tmp_group[0]


    #true_title = tmp_group.find_all('div', attrs = {'class': 'article__header__title'})
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
        # raise
    else:
        #if len(tmp_group) > 1:
           # snd_text = 
        tmp_group = tmp_group[0]

    overview = tmp_group.find_all('div', attrs={'class': 'article__text__overview'})
    if len(overview) == 0:
        #print(f'{inp_id} something went wrong. overview {len(overview)}')
        #print(overview, true_category, true_title)
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



    #print(true_category, true_title, tmp_imgs, '\n')
    #print(overview, '\n')
    #print(text)
    #print(text[0].text)
    return (true_category, true_title, tmp_imgs, overview, len(text.split()), two_articles, text) #snd_header, snd_text

Фоторепортаж, Фотогалерея, Главное за день, ЧЭЗ, видео, прямая трансляция
# In[12]:


#df_train[df_train.document_id == '626e564d9a79471a3cd5de65ZM028L7kQ1mVIZAB30bTEA'].title.values
df_test[df_test.document_id == '620a7cbf9a79471a9c6ace46aMuqupFlTxSsa5P6zHzaEQ'].title.values


# In[13]:


df_train[df_train.document_id == '6210c3939a7947e58a257424iqcwqgm9QXShvP0aU1iVQQ']


# In[14]:


#620d1f0c9a794724696a95e7igKOAeqwSo6yt6MHdm1JNA something went wrong. text
#6210c3939a7947e58a257424iqcwqgm9QXShvP0aU1iVQQ something went wrong. text
#6253d6f59a7947a4e4819c4eXWVPJk6OTUOpRafSX6B9lQ something went wrong. text
#6278ac619a79475802c0682aE2s6qP24SsCW5dkYJMTCkA something went wrong. text
#61f954049a79479310c59dcf10GpiD-VRHCC631Hkl2Y4Q something went wrong. text
#61fd4d109a794786c8d4dc59COhaOYZzT8qrDBW6plnsDw something went wrong. text
#62487fc99a7947476b4c938bcZ5KFPtbQF6EmK9oG7vWMA something went wrong. text
#626459ea9a79477bae9c49313bq0StmMT2uyeuDZmZKmyA something went wrong. text
#626e36de9a794710fdef04c1-SCC98EoT7u11HsvVQ7rIQ something went wrong. text
#623b031f9a79474a28a2ce99AL-9lSRYR46n0_5tw7Bd0A something went wrong. text
#624fdb999a79471adecb2b79t0GMIFtZQv-nadA-xJiaYg something went wrong. text
#61fbfa689a79470784c13d75W4OgaC-ySTiD34lTz6Sj9g something went wrong. text
#626e564d9a79471a3cd5de65ZM028L7kQ1mVIZAB30bTEA something went wrong. text



# 620fef1c9a7947b2de6c18f6rwBF3WoeQbm1jgCkN6cGAQ something went wrong. text
# 620e76c79a794723bf70e50bSoTq5ec2Raq3SK7ZVq8WcQ something went wrong. text
# 628201039a7947e9fde98653Qc5leGRRTO2feOAlfm5BwA something went wrong. text
# 620a7cbf9a79471a9c6ace46aMuqupFlTxSsa5P6zHzaEQ something went wrong. text


# In[ ]:





# In[15]:


df_train['tmp'] = df_train.document_id.progress_apply(get_article_info)
df_test['tmp'] = df_test.document_id.progress_apply(get_article_info)


# In[16]:


#(true_category, true_title, tmp_imgs, overview, len(text.split()), two_articles, )


# In[17]:


df_train['true_category'] = df_train.tmp.apply(lambda x: x[0])
df_train['true_title'] = df_train.tmp.apply(lambda x: x[1])
df_train['nimgs'] = df_train.tmp.apply(lambda x: x[2])
df_train['overview'] = df_train.tmp.apply(lambda x: x[3])
df_train['text_len'] = df_train.tmp.apply(lambda x: x[4])
df_train['two_articles'] = df_train.tmp.apply(lambda x: x[5])


df_test['true_category'] = df_test.tmp.apply(lambda x: x[0])
df_test['true_title'] = df_test.tmp.apply(lambda x: x[1])
df_test['nimgs'] = df_test.tmp.apply(lambda x: x[2])
df_test['overview'] = df_test.tmp.apply(lambda x: x[3])
df_test['text_len'] = df_test.tmp.apply(lambda x: x[4])
df_test['two_articles'] = df_test.tmp.apply(lambda x: x[5])


df_train.drop(['tmp'], axis = 1, inplace = True)
df_test.drop(['tmp'], axis = 1, inplace = True)


# In[22]:


def decode_parsing(inp_df):
    
    inp_df['true_category'] = inp_df.tmp.apply(lambda x: x[0])
    inp_df['true_title'] = inp_df.tmp.apply(lambda x: x[1])
    inp_df['nimgs'] = inp_df.tmp.apply(lambda x: x[2])
    inp_df['overview'] = inp_df.tmp.apply(lambda x: x[3])
    inp_df['text_len'] = inp_df.tmp.apply(lambda x: x[4])
    inp_df['two_articles'] = inp_df.tmp.apply(lambda x: x[5])
    
    inp_df.drop(['tmp'], axis = 1, inplace = True)
    
    return inp_df


# In[ ]:





# In[18]:


#df_train[(df_train.true_title.apply(lambda x: x.endswith('COVID')))]['true_title'].values
#df_train[(df_train.true_title.apply(lambda x: 'телеканале РБК' in x))]['true_title'].values


# In[19]:


# Прямая трансляция, Фоторепортаж, Фотогалерея, Видео, телеканале РБК, Инфографика endswith


# In[20]:


#df_train[df_train.text_len == 0].true_title.values


# In[ ]:





# In[ ]:





# In[21]:


df_train.to_csv(os.path.join(DIR_DATA, 'train_extended.csv'), index = False)
df_test.to_csv(os.path.join(DIR_DATA, 'test_extended.csv'), index = False)


# In[ ]:




