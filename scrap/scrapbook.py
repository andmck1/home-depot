
# coding: utf-8

# # Scrapbook for Home-depot Question

# ## Imports

# In[1]:


# Data Wrangling
import numpy as np
import pandas as pd


# In[94]:


# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[80]:


# Misc
import os
import re
from pprint import pprint as pp


# In[190]:


# Machine Learning
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


# ## Config

# In[5]:


data_path = "../../scrap/KAG_home-depot/"


# In[6]:


stemmer = SnowballStemmer('english')


# In[ ]:


nlp = spacy.load('en_core_web_lg')


# In[159]:


nlp_tag = spacy.load('en', disable=['parser', 'ner'])


# ## Functions

# In[8]:


def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])


# In[9]:


def str_common_word(str1, str2):
    return sum(int(str2.find(word)>=0) for word in str1.split())


# In[10]:


def process_data(df):
    df['product_title'] = df['product_title'].apply(lambda x: str_stemmer(x))
    df['search_term'] = df['search_term'].apply(lambda x: str_stemmer(x))
    df['product_description'] = df['product_description'].apply(lambda x: str_stemmer(x))
    df['query_len'] = df['search_term'].apply(lambda x: len(x.split()))
    df['product_info'] = df['search_term']+"\t"+df['product_title']+"\t"+df['product_description']
    df['word_in_title'] = df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
    df['word_in_description'] = df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
    return df


# In[ ]:


def get_pos_counts(search_terms_list):
    st = nlp(search_terms_list[0])
    pos_st1 = pd.Series([x.pos_ for x in st]).value_counts()
    del st
    st = nlp(search_terms_list[1])
    pos_st2 = pd.Series([x.pos_ for x in st]).value_counts()
    del st
    pos_tot = (pos_st1 + pos_st2).sort_values()


# In[ ]:


# # Plot
# ax = pos_tot.plot(kind='bar')
# ax.set_title('Count of POS Tags', )
# ax.set_xlabel('POS Tag')
# ax.set_ylabel('Count')
# plt.savefig('pos_tags_bar.png', dpi=500, bbox_inches="tight")


# In[142]:


def lemm(word, nlp):
    return nlp(word)[0].lemma_


# ## Import data

# In[11]:


os.listdir('../../scrap/')
csvs = [f for f in os.listdir(data_path) if re.search(".csv$", f)]
print(csvs)
data_dict = {}
for csv in csvs:
    data_dict[csv.split()[0]] = pd.read_csv(data_path+csv, encoding='latin1')
test_df = data_dict['test.csv']
train_df = data_dict['train.csv']
prod_desc = data_dict['product_descriptions.csv']
attributes = data_dict['attributes.csv']
sample_sub = data_dict['sample_submission.csv']


# ## High-Level Overview

# ### Test Data

# In[12]:


test_df.head()


# In[13]:


test_df.info()


# ### Train Data

# In[14]:


train_df.head()


# In[15]:


train_df.info()


# ### Sample Sub

# In[16]:


sample_sub.head()


# In[17]:


sample_sub.info()


# ### Attributes of Products

# In[18]:


attributes.head()


# In[19]:


attributes.info()


# ### Product Descriptions

# In[20]:


prod_desc.head()


# In[21]:


prod_desc.info()


# ## Investigation

# Let start by using moomin's sklearn random forest kernel on Kaggle to get a starting point

# In[32]:


attr = attributes.dropna(how='all')
attr.loc[:,'product_uid'] = attr.loc[:,'product_uid'].apply(lambda x: int(x))
attr['name'] = attr['name'].apply(lambda x: 'Multi' if "Bullet" in x else x)
attr = attr.groupby(['product_uid','name']).apply(lambda x: '\n'.join(x['value'].dropna())).reset_index(name='value').set_index('product_uid')


# In[22]:


df_train = pd.merge(train_df, prod_desc, how='left', on='product_uid').drop('id', axis=1)
df_test = pd.merge(test_df, prod_desc, how='left', on='product_uid').drop('id', axis=1)


# In[23]:


df_train[df_train['relevance']==3.0].head()


# In[24]:


# Get docs
pt_doc = nlp(df_train.iloc[0,1])
st_doc = nlp(df_train.iloc[0,2])
pd_doc = nlp(df_train.iloc[0,4])


# In[29]:


df_train['search_term'].apply(lambda x: len(x.split())).mean()


# In[52]:


uid = 100002
print(df_train[(df_train['relevance']==3.0) & (df_train['product_uid']==uid)].head()['product_title'].values[0])
print()
print(df_train[(df_train['relevance']==3.0) & (df_train['product_uid']==uid)].head()['search_term'].values[0])
print()
print(df_train[(df_train['relevance']==3.0) & (df_train['product_uid']==uid)].head()['product_description'].values[0])
print()
pp(attr.loc[uid, 'name'])
pp(attr.loc[uid, 'value'].values)


# In[33]:


print([x for x in pd_doc if not x.is_stop])


# In[67]:


search_terms = ' '.join(df_train['search_term'].values)


# In[70]:


l = len(search_terms)
search_terms_list = [
    search_terms[0:int(l/2)],
    search_terms[int(l/2):]
]


# In[126]:


words_st = ' '.join(df_train['search_term'].values).split()


# In[156]:


len(words_st)/5000 * 9.64 / 60


# In[158]:


len(words_st)/5000 * 32 / 60


# In[165]:


import time
start = time.time()
# list((map(lambda x: lemm(x, nlp_tag), words_st[:5000])))
stems = list(map(lambda x: (stemmer.stem(x),), words_st))
print(time.time()-start)


# Okay. That was decent. However, let's work on this and do better!

# Attributes wasn't used...I think there must be something here. Also only stems were taken...can we not extract keywords?

# In[191]:


nltk.pos_tag(nltk.word_tokenize(' '.join(attr.loc[100001, 'value'].values)))

