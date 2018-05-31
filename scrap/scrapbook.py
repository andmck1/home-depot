
# coding: utf-8

# # Scrapbook for Home-depot Question

# ## Imports

# In[1]:


# Data Wrangling
import numpy as np
import pandas as pd


# In[2]:


# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().magic('matplotlib inline')


# In[3]:


# Misc
import os
import re
from pprint import pprint as pp
import logging


# In[4]:


# Machine Learning
import nltk
import spacy
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


# ## Config

# In[5]:


logging.basicConfig(filename='run.log', level=logging.INFO)


# In[6]:


pd.options.mode.chained_assignment = None


# In[7]:


data_path = "../../scrap/KAG_home-depot/"


# In[8]:


stemmer = SnowballStemmer('english')


# In[9]:


nlp = spacy.load('en')


# In[10]:


nlp_tag = spacy.load('en', disable=['parser', 'ner'])


# ## Functions

# In[11]:


def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])


# In[12]:


def str_common_word(str1, str2):
    return sum(int(str2.find(word)>=0) for word in str1.split())


# In[13]:


def process_data(df):
    df['product_title'] = df['product_title'].apply(lambda x: str_stemmer(x))
    df['search_term'] = df['search_term'].apply(lambda x: str_stemmer(x))
    df['product_description'] = df['product_description'].apply(lambda x: str_stemmer(x))
    df['query_len'] = df['search_term'].apply(lambda x: len(x.split()))
    df['product_info'] = df['search_term']+"\t"+df['product_title']+"\t"+df['product_description']
    df['word_in_title'] = df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
    df['word_in_description'] = df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
    return df


# In[14]:


def get_pos_counts(search_terms_list):
    st = nlp(search_terms_list[0])
    pos_st1 = pd.Series([x.pos_ for x in st]).value_counts()
    del st
    st = nlp(search_terms_list[1])
    pos_st2 = pd.Series([x.pos_ for x in st]).value_counts()
    del st
    pos_tot = (pos_st1 + pos_st2).sort_values()


# In[15]:


# # Plot
# ax = pos_tot.plot(kind='bar')
# ax.set_title('Count of POS Tags', )
# ax.set_xlabel('POS Tag')
# ax.set_ylabel('Count')
# plt.savefig('pos_tags_bar.png', dpi=500, bbox_inches="tight")


# In[16]:


def lemm(word, nlp):
    return nlp(word)[0].lemma_


# ## Import data

# In[17]:


logging.info("Importing Data...")
os.listdir('../../scrap/')
csvs = [f for f in os.listdir(data_path) if re.search(".csv$", f)]
data_dict = {}
for csv in csvs:
    data_dict[csv.split()[0]] = pd.read_csv(data_path+csv, encoding='latin1')
test_df = data_dict['test.csv']
train_df = data_dict['train.csv']
prod_desc = data_dict['product_descriptions.csv']
attributes = data_dict['attributes.csv']
sample_sub = data_dict['sample_submission.csv']


# ## Data Handling

# In[18]:


logging.info("Preparing Data...")
attr = attributes.dropna(how='all')
attr[['name','value']] = attr[['name','value']].fillna('')
attr['product_uid'] = attr['product_uid'].apply(lambda x: int(x))
attr['name'] = attr['name'].apply(lambda x: '' if "Bullet" in x else x)
attr['Attributes'] = attr['name'] + '\t' + attr['value'] + '\n'
attr = attr.drop(['name','value'], axis=1).groupby('product_uid').sum().reset_index()
df_train = pd.merge(train_df, prod_desc, how='left', on='product_uid').drop('id', axis=1).merge(attr, on='product_uid')


# In[19]:


df_train.head()


# ## Investigation

# In[20]:


df_train['relevance'].hist(bins=8)


# In[22]:


df_train['search_term'].apply(lambda x: len(x.split())).mean()


# In[ ]:


words_st = ' '.join(df_train['search_term'].values).split()


# In[ ]:


len(words_st)/5000 * 9.64 / 60


# In[ ]:


len(words_st)/5000 * 32 / 60


# In[ ]:


import time
start = time.time()
# list((map(lambda x: lemm(x, nlp_tag), words_st[:5000])))
stems = list(map(lambda x: (stemmer.stem(x),), words_st))
print(time.time()-start)


# Okay. That was decent. However, let's work on this and do better!

# Attributes wasn't used...I think there must be something here. Also only stems were taken...can we not extract keywords?

# In[ ]:


nltk.pos_tag(nltk.word_tokenize(' '.join(attr.loc[100001, 'value'].values)))


# In[ ]:


attr.head()


# In[ ]:


master.head()


# In[ ]:


master['query_len'] = master['search_term'].apply(lambda x: len(x.split()))


# In[ ]:


master['query_len'].hist(bins=15)


# In[ ]:


master[['query_len','relevance']].plot(kind='scatter', x='relevance', y='query_len', alpha=0.4)


# In[ ]:


logging.info("NLP brah...")
# import time
# start = time.time()
master_nlp = master.loc[:, ['product_title','search_term', 'product_description','Attributes']].applymap(nlp_tag)
# print("Time to complete is: {}".format((time.time()-start)*len(master)/(500*60)))


# In[ ]:


master_nlp.to_pickle(data_path+'master_nlp.pickle')


# In[ ]:


# Remove stop characters, punctuation, 

# Lemmatise

