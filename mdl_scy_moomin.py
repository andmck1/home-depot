# # Scrapbook for Home-depot Question

# Imports

## Data Wrangling
import numpy as np
import pandas as pd

## Misc
import os
import re
from pprint import pprint as pp

## Machine Learning
import spacy
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Config

data_path = "../scrap/KAG_home-depot/"
nlp = spacy.load('en_core_web_sm')


# Functions

def str_stemmer(s):
    return " ".join([nlp(word)[0].lemma_ for word in s.lower().split()])

def str_common_word(str1, str2):
    return sum(int(str2.find(word)>=0) for word in str1.split())

def process_data(df):
    df['product_title'] = df['product_title'].apply(lambda x: str_stemmer(x))
    df['search_term'] = df['search_term'].apply(lambda x: str_stemmer(x))
    df['product_description'] = df['product_description'].apply(lambda x: str_stemmer(x))
    df['query_len'] = df['search_term'].apply(lambda x: len(x.split()))
    df['product_info'] = df['search_term']+"\t"+df['product_title']+"\t"+df['product_description']
    df['word_in_title'] = df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
    df['word_in_description'] = df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
    return df


# Import data

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


# Process data

df_train = pd.merge(train_df, prod_desc, how='left', on='product_uid').drop('id', axis=1)
df_test = pd.merge(test_df, prod_desc, how='left', on='product_uid').drop('id', axis=1)
df_train = process_data(df_train)
df_test = process_data(df_test)


# ML Algo

X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['product_uid','relevance'], axis=1), df_train['relevance'], test_size=0.33, random_state=42)
rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# Print metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
