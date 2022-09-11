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


class TitleEmb:
    def __init__(self, inp_model_name: str, inp_model_folder: str, inp_max_length: int, inp_npca_components: int) -> None:
        """
        Инициализация класса
        args:
            inp_model_name   - название модели на huggingface
            inp_model_folder - папка модели
            inp_max_length   - максимальная длинна текста (заголовка) для обработки
            inp_npca_components - длина эмбеддинга после PCA (кол-во компонент)
        return:
        """
        self.pre_trained_model_name   = inp_model_name
        self.pre_trained_model_folder = inp_model_folder
        self.max_length = inp_max_length
        self.npca_components = inp_npca_components
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.pre_trained_model_name)
        self.model = AutoModel.from_pretrained(self.pre_trained_model_name)     
        _ = self.model.cpu()

        self.pca = PCA(n_components = self.npca_components)
        

    #Mean Pooling - Take attention mask into account for correct averaging    
    def mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Усреднение с учетом маски (более точное)
        args:
            model_output - предсказанный эмбеддинг
            attention_mask - масска внимания для уточнения пулинга
        return:
            усредненный эмбеддинг с учетом маски внимания
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask
    
    
    
    def ttl_to_emb(self, inp_text: str) -> np.ndarray:
        """
        Преобразование заголовка к эмбеддингу
        args:
            inp_text - текст (заголовок), для преобразования в эмбеддинги
        return:
            np.ndarray - эмбеддинг входного текста
        """
        encoded_input = self.tokenizer(inp_text, padding = True, truncation = True, max_length = self.max_length, return_tensors = 'pt')

        #Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        #print('model_output', type(model_output))
        #print('encoded_input', type(encoded_input['attention_mask']))
        #print('sentence_embeddings', type(sentence_embeddings))
        return sentence_embeddings[0].cpu().detach().numpy()



    def make_title_emb_features(self, inp_df: pd.DataFrame, train_pca: bool) -> pd.DataFrame:
        """
        Добавление эмбеддингов заголовков к заданному DataFram'у
        args:
            inp_df - входной pd.DataFrame для преобразования
            train_pca - необходимо ли обучать PCA или только преобразовывать
        return:
            pd.DataFrame - исходный DataFrame расширенный эмбеддингами заголовков
        """
        inp_df = inp_df[['document_id', 'true_title']]
        #inp_df.loc[:, 'ttl_emb'] = inp_df.true_title.progress_apply(lambda x: self.ttl_to_emb(x))
        inp_df['ttl_emb'] = inp_df.true_title.progress_apply(lambda x: self.ttl_to_emb(x))
        
        if train_pca:
            self.pca.fit(inp_df.ttl_emb.to_list())
            print('fitting pca')
            
        col_names = [f'tt_emb{idx}' for idx in range(self.npca_components)]
        emb_train = pd.DataFrame(self.pca.transform(inp_df.ttl_emb.to_list()), columns = col_names)
        
        inp_df = pd.concat([inp_df, emb_train], axis=1)
        inp_df.drop('ttl_emb', axis = 1, inplace = True)
    
        return inp_df


# In[13]:


#PRE_TRAINED_MODEL_NAME = 'blanchefort/rubert-base-cased-sentiment-rurewiews'
#MODEL_FOLDER = 'ru-blanchefort-rurewiews2'

#'DeepPavlov/rubert-base-cased-sentence'
#'sberbank-ai/sbert_large_mt_nlu_ru'

#PRE_TRAINED_MODEL_NAME = 'DeepPavlov/rubert-base-cased-sentence'
#MODEL_FOLDER = 'rubert-base-cased-sentence'

PRE_TRAINED_MODEL_NAME = 'sberbank-ai/sbert_large_mt_nlu_ru'
MODEL_FOLDER = 'sbert_large_mt_nlu_ru'


MAX_LENGTH = 24

PCA_COMPONENTS = 64


# In[14]:


te = TitleEmb(PRE_TRAINED_MODEL_NAME, MODEL_FOLDER, MAX_LENGTH, PCA_COMPONENTS)


# In[15]:


print('before ', df_train.shape, df_test.shape)
df_train = te.make_title_emb_features(df_train, True)
df_test  = te.make_title_emb_features(df_test,  False)
print('after  ', df_train.shape, df_test.shape)


# In[16]:


#df_train2 = te.make_title_emb_features(df_train[:100], True)


# In[ ]:





# In[ ]:





# In[ ]:





# Сохраняем только эмбеддинги, без остальных признаков

# In[24]:


df_train.to_csv(os.path.join(DIR_DATA, f'ttl_cln_emb_train_{MODEL_FOLDER}_{MAX_LENGTH}_pca{PCA_COMPONENTS}.csv'), index = False)


# In[31]:


df_test.to_csv(os.path.join(DIR_DATA, f'ttl_cln_emb_test_{MODEL_FOLDER}_{MAX_LENGTH}_pca{PCA_COMPONENTS}.csv'), index = False)


# In[ ]:





# In[32]:


#ray.shutdown()


# In[33]:


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


# In[ ]:




