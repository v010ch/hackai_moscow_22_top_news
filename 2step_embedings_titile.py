#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle as pkl

import numpy as np
import pandas as pd

import re
from tqdm import tqdm
from tqdm.auto import tqdm  # for notebooks
tqdm.pandas()


# In[2]:


import torch
#$from transformers import AutoModelForSequenceClassification
#from transformers import BertTokenizerFast
#from transformers import AutoTokenizer
#from transformers import Trainer, TrainingArguments

from transformers import AutoTokenizer, AutoModel


# In[ ]:





# In[3]:


DIR_DATA  = os.path.join(os.getcwd(), 'data')


# In[4]:


#MUSE, sbert_large_mt_nlu_ru и rubert-base-cased-sentence


# ## Prepare data

# In[5]:


df_train = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'))#, index_col= 0)
df_test  = pd.read_csv(os.path.join(DIR_DATA, 'test.csv'))#, index_col= 0)


# In[6]:


# sberbank-ai/sbert_large_mt_nlu_ru       1024  1.71Gb
# DeepPavlov/rubert-base-cased-sentence   768   0.7Gb
# DeepPavlov/rubert-base-cased-conversational  768
# DeepPavlov/rubert-base-cased            768
# sberbank-ai/sbert_large_nlu_ru          1024  1.71Gb


# In[ ]:





# In[7]:


# should try and without it
clean_text = lambda x:' '.join(re.sub('\n|\r|\t|[^а-я]', ' ', x.lower()).split())


# In[8]:


x = clean_text(df_train.title[0])


# In[9]:


x


# In[10]:


#dir(model)


# ## Load model

# In[11]:


#PRE_TRAINED_MODEL_NAME = 'blanchefort/rubert-base-cased-sentiment-rurewiews'
#MODEL_FOLDER = 'ru-blanchefort-rurewiews2'

#'DeepPavlov/rubert-base-cased-sentence'
#'sberbank-ai/sbert_large_mt_nlu_ru'

PRE_TRAINED_MODEL_NAME = 'DeepPavlov/rubert-base-cased-sentence'
MODEL_FOLDER = 'rubert-base-cased-sentence'


MAX_LENGTH = 24


# In[12]:


tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)


#tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#train_tokens = tokenizer(list(train.values), truncation=True, padding=True, max_length=MAX_LENGTH)
#test_tokens = tokenizer(list(test.values), truncation=True, padding=True, max_length=MAX_LENGTH)

#model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME,) 


# In[13]:


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# In[ ]:





# In[14]:


def ttl_to_emb(inp_text):
    encoded_input = tokenizer(inp_text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')

    #Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    return sentence_embeddings[0].cpu().detach().numpy()


# ## Make embedings for titles. Train

# In[15]:


df_train = df_train[['document_id', 'title']]


# In[16]:


df_train['ttl_emb'] = df_train.title.progress_apply(lambda x: ttl_to_emb(x))


# In[17]:


col_names = [f'tt_emb{idx}' for idx in range(df_train.ttl_emb[0].shape[0])]
emb_train = pd.DataFrame(df_train.ttl_emb.to_list(), columns = col_names)


# In[18]:


df_train = pd.concat([df_train, emb_train], axis=1)


# In[19]:


df_train.drop('ttl_emb', axis = 1, inplace = True)


# In[20]:


df_train.head(3)


# In[21]:


df_train.to_csv(os.path.join(DIR_DATA, f'ttl_emb_train_{MODEL_FOLDER}_{MAX_LENGTH}.csv'), index = False)


# In[ ]:





# In[ ]:





# ## Same with test

# In[22]:


df_test = df_test[['document_id', 'title']]


# In[23]:


df_test['ttl_emb'] = df_test.title.progress_apply(lambda x: ttl_to_emb(x))


# In[24]:


col_names = [f'tt_emb{idx}' for idx in range(df_test.ttl_emb[0].shape[0])]
emb_test = pd.DataFrame(df_test.ttl_emb.to_list(), columns = col_names)


# In[25]:


df_test = pd.concat([df_test, emb_test], axis=1)


# In[26]:


df_test.drop('ttl_emb', axis = 1, inplace = True)


# In[27]:


df_test.shape


# In[28]:


df_test.to_csv(os.path.join(DIR_DATA, f'ttl_emb_test_{MODEL_FOLDER}_{MAX_LENGTH}.csv'), index = False)


# In[ ]:





# In[ ]:




