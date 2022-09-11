#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[2]:


get_ipython().run_line_magic('watermark', '')


# In[3]:


import time
notebookstart = time.time()


# In[4]:


import os
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import multiprocessing as mp

import re
from tqdm import tqdm
from tqdm.auto import tqdm
tqdm.pandas()


# In[5]:


# os.environ["TOKENIZERS_PARALLELISM"] = 'true'


# In[6]:


import torch
from transformers import AutoTokenizer, AutoModel


# In[7]:


#from functools import partial
#from embtitile import embtitile as et


# In[8]:


#import ray
#ray.init()


# In[ ]:





# Переменные

# In[9]:


DIR_DATA  = os.path.join(os.getcwd(), 'data')


# In[ ]:





# ## Загружаем и подготавливаем данные

# In[10]:


#'_extended' после парсинга данных с РБК и извлечения данных из спарсенных страниц
df_train = pd.read_csv(os.path.join(DIR_DATA, 'train_extended.csv'))
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test_extended.csv'))


# In[11]:


#имя            размерность выходного вектора   вес модели
# sberbank-ai/sbert_large_mt_nlu_ru       1024  1.71Gb
# DeepPavlov/rubert-base-cased-sentence   768   0.7Gb
# DeepPavlov/rubert-base-cased-conversational  768
# DeepPavlov/rubert-base-cased            768
# sberbank-ai/sbert_large_nlu_ru          1024  1.71Gb


# In[ ]:





# ## Загружаем модель

# In[12]:


#PRE_TRAINED_MODEL_NAME = 'blanchefort/rubert-base-cased-sentiment-rurewiews'
#MODEL_FOLDER = 'ru-blanchefort-rurewiews2'

#'DeepPavlov/rubert-base-cased-sentence'
#'sberbank-ai/sbert_large_mt_nlu_ru'

#PRE_TRAINED_MODEL_NAME = 'DeepPavlov/rubert-base-cased-sentence'
#MODEL_FOLDER = 'rubert-base-cased-sentence'

PRE_TRAINED_MODEL_NAME = 'sberbank-ai/sbert_large_mt_nlu_ru'
MODEL_FOLDER = 'sbert_large_mt_nlu_ru'


MAX_LENGTH = 24


# In[13]:


tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

_ = model.cpu()


# In[14]:


#dir(tokenizer)


# In[15]:


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return sum_embeddings / sum_mask


# In[16]:


def ttl_to_emb(inp_text: str) -> np.ndarray:
    
    encoded_input = tokenizer(inp_text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')

    #Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    return sentence_embeddings[0].cpu().detach().numpy()


# In[ ]:





# In[ ]:





# ## Делаем эмбеддинга из заголовков. Трейн

# In[17]:


df_train = df_train[['document_id', 'true_title']]


# In[18]:


df_train['ttl_emb'] = df_train.true_title.progress_apply(lambda x: ttl_to_emb(x))

col_names = [f'tt_emb{idx}' for idx in range(df_train.ttl_emb[0].shape[0])]
emb_train = pd.DataFrame(df_train.ttl_emb.to_list(), columns = col_names)
# In[19]:


PCA_COMPONENTS = 64


# In[20]:


get_ipython().run_cell_magic('time', '', "ttl_pca = PCA(n_components = PCA_COMPONENTS)\nttl_pca.fit(df_train.ttl_emb.to_list())\n\ncol_names = [f'tt_emb{idx}' for idx in range(PCA_COMPONENTS)]\nemb_train = pd.DataFrame(ttl_pca.transform(df_train.ttl_emb.to_list()), columns = col_names)")


# In[21]:


df_train = pd.concat([df_train, emb_train], axis=1)


# In[22]:


df_train.drop('ttl_emb', axis = 1, inplace = True)


# In[23]:


df_train.head(3)


# Сохраняем только эмбеддинги, без остальных признаков

# In[24]:


df_train.to_csv(os.path.join(DIR_DATA, f'ttl_cln_emb_train_{MODEL_FOLDER}_{MAX_LENGTH}_pca{PCA_COMPONENTS}.csv'), index = False)


# In[ ]:





# In[ ]:





# ## Выполняем тоже с тестом

# In[25]:


df_test = df_test[['document_id', 'true_title']]


# In[26]:


df_test['ttl_emb'] = df_test.true_title.progress_apply(lambda x: ttl_to_emb(x))


# Сокращаем размерность

# In[27]:


#col_names = [f'tt_emb{idx}' for idx in range(df_test.ttl_emb[0].shape[0])]
emb_test = pd.DataFrame(ttl_pca.transform(df_test.ttl_emb.to_list()), columns = col_names)
#emb_test = pd.DataFrame(df_test.ttl_emb.to_list(), columns = col_names)


# In[28]:


df_test = pd.concat([df_test, emb_test], axis=1)


# In[29]:


df_test.drop('ttl_emb', axis = 1, inplace = True)


# In[30]:


df_test.shape


# Сохраняем только эмбеддинги, без остальных признаков

# In[31]:


df_test.to_csv(os.path.join(DIR_DATA, f'ttl_cln_emb_test_{MODEL_FOLDER}_{MAX_LENGTH}_pca{PCA_COMPONENTS}.csv'), index = False)


# In[ ]:





# In[32]:


#ray.shutdown()


# In[33]:


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


# In[ ]:




