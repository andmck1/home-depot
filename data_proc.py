# Author: Alastair Hamilton
# Date: May/June 2018
# Title: Model for Home-depot Kaggle Competition


# Imports

## Data Wrangling
import numpy as np
import pandas as pd

## Misc
import os
import re
from pprint import pprint as pp
import time

## Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

## NLP
import spacy

## ML
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Config

# # Pandas error display OFF
pd.options.mode.chained_assignment = None

# # Set path to data
data_path = "./data/"

# # Create spaCy nlp tagger
nlp_tag = spacy.load('en', disable=['parser', 'ner'])

# # Processing features
proc_feat = ['search_term', 'product_title', 'product_description', 'attributes']

# Functions

# Remove punctuation from column of dataframe
def rmv_punc(df, col):
    df.loc[:, col] = df.loc[:,col].apply(lambda x: tuple(filter(lambda y: not y.is_punct, x)))
    return df

# Remove stop words and punctuation
def rmv_stoppunc(s):
    return s.apply(lambda x: tuple(filter(lambda y: not (y.is_stop or y.is_punct), x)))

# Apply function on rows of data frame (2 cols max)
def func_row(df, func):
    return df.apply(lambda row: func(row[0], row[1]), axis=1)

# Find number of words in one document (doc1) that are in another document (doc2)
def common_words_doc(doc1, doc2):
    tot = 0
    for w1 in doc1:
        for w2 in doc2:
#             if w1.lemma_ == w2.lemma_:
#                 tot += 1
#                 break
            if w2.lemma_.find(w1.lemma_) >= 0:
                tot += 1
                break
    return tot


# Get starting time for script
stime = time.time()


# -------------------------
print('\n-------------------------')

# Import data
print('Importing data...')

# # Get all zipped files in data path
zips = [f for f in os.listdir(data_path) if re.search(".zip$", f)]

# # Unzip all files and put into dictionary, keyed by file stem
data_dict = {}
for zipped in zips:
    print('- Importing {}...'.format(zipped))
    data_dict[zipped.split('.')[0]] = pd.read_csv(data_path+zipped, compression='zip', encoding='latin1')

# # Set dataframe to piece in data dictionary
train_df = data_dict['train']
prod_desc = data_dict['product_descriptions']
attributes = data_dict['attributes']

# # Clean up
del data_dict

# # Time
print('Took {:.2f} minutes.'.format((time.time()-stime)/60))

# -------------------------
print('-------------------------')

# Time
itime = time.time()
      
# Process data
print("Processing data...")

# # Process attributes data
print("- Handling attributes data...")

# # # Deal with N/As in attributes data (drop empty records and fill in name and values with empty string)
attr = attributes.dropna(how='all')
attr[['name','value']] = attr[['name','value']].fillna('')

# # # Ensure UID is int
attr['product_uid'] = attr['product_uid'].apply(lambda x: int(x))

# # # If "bullet" in attribute name then asserting name is meaningless - make an empty string
attr['name'] = attr['name'].apply(lambda x: '' if "Bullet" in x else x)

# # # Group name and value in attributes into single column, separated by a tab and ending in newline (for grouping stage next)
attr['attributes'] = attr['name'] + '\t' + attr['value'] + '\n'

# # # Drop name and values, groupby UID and sum grouped values, reset index...
# # # ...(ie. all attributes in single cell now, separated by newlines as set up above)
attr = attr.drop(['name','value'], axis=1).groupby('product_uid').sum().reset_index()

# # Create master data frame
print("- Creating master data frame...")

# # # Merge all data into one master dataframe by merging descriptions and attributes onto training data on UID...
# # # ...Fill any NAs with empty string
data = pd.merge(train_df, prod_desc, how='left', on='product_uid').drop('id', axis=1).merge(attr, on='product_uid', how='left').fillna('')

# # # Finally create a master index column, which will be used to reference individual search terms
data = data.reset_index().drop('index', axis=1).reset_index()

# # Clean up
del train_df
del prod_desc
del attr

# # Time
print('Took {:.2f} minutes.'.format((time.time()-itime)/60))


# -------------------------
print("-------------------------")

# Time
itime = time.time()

# Feature Generation
print("Generating features...")

# # NLP
print("- Applying spaCy NLP processor to all string features...")
for feat in proc_feat:
    print('-- Applying to {}'.format(feat))
    data[feat] = data[feat].apply(nlp_tag)

# # Len of query
print('- Creating length of query column...')
data['q_len'] = data['search_term'].apply(lambda x: len(x))

# # Get common words between query and returned product title
print('- Creating query-title common words column...')
data['com_title'] = func_row(data[['search_term', 'product_title']], common_words_doc)

# # Get common words between query and returned product description
print('- Creating query-description common words column...')
data['com_desc'] = func_row(data[['search_term', 'product_description']], common_words_doc)

# # Get common words between query and returned product description
print('- Creating query-attributes common words column...')
data['com_attr'] = func_row(data[['search_term', 'attributes']], common_words_doc)

# # Clean up
data = data.drop(proc_feat, axis=1)

# # Write data to file
print('- Writing data to file...')
data.to_csv(data_path+'features.csv')

# # Time
print('Took {:.2f} minutes.'.format((time.time()-itime)/60))

# Finish up
pp(data.head())
print('Script took a total of {:.2f} minutes'.format((time.time()-stime)/60))
print('Done!')
