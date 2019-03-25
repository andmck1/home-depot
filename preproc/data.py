# Author: Alastair Hamilton
# Date: April 2019
# Title: Model for Home-depot Kaggle Competition

# ---------------
# IMPORTS -------
# ---------------
# Library
import numpy as np
import pandas as pd
import spacy
import os
import re

# Local
from .lang import *


# ---------------
# CONFIG --------
# ---------------

# Pandas error display OFF
pd.options.mode.chained_assignment = None

# Set path to data
data_path = "./data/"

# Create spaCy nlp tagger
nlp_tag = spacy.load('en', disable=['parser', 'ner'])

# Processing features
proc_feat = ['search_term',
             'product_title',
             'product_description',
             'attributes']


# ---------------
# FUNCTIONS -----
# ---------------


# ---------------
# CLASSES -------
# ---------------

class Data:
    def __init__(self):
        # Import data
        print('Importing data...')

        # Get all zipped files in data path
        zips = [f for f in os.listdir(data_path) if re.search(".zip$", f)]

        # Unzip all files and put into dictionary, keyed by file stem
        data_dict = {}
        for zipped in zips:
            print('- Importing {}...'.format(zipped))
            data_dict[zipped.split('.')[0]] = pd.read_csv(data_path+zipped, compression='zip', encoding='latin1')

        # Set dataframe to piece in data dictionary
        train_df = data_dict['train']
        prod_desc = data_dict['product_descriptions']
        attributes = data_dict['attributes']

        # Clean
        self.preproc = self.clean(train_df, prod_desc, attributes)

        # Feature Generation
        self.data = self.gen_features(self.preproc)

    def clean(self, train_df, prod_desc, attributes):
        # Deal with N/As in attributes data (drop empty records and fill in
        # ...name and values with empty string)
        attr = attributes.dropna(how='all')
        attr[['name', 'value']] = attr[['name', 'value']].fillna('')

        # Ensure UID is int
        attr['product_uid'] = attr['product_uid'].apply(lambda x: int(x))

        # If "bullet" in attribute name then asserting name is meaningless -
        # ...make an empty string
        attr['name'] = attr['name'].apply(lambda x: '' if "Bullet" in x else x)

        # Group name and value in attributes into single column, separated by
        # ...a tab and ending in newline (for grouping stage next)
        attr['attributes'] = attr['name'] + '\t' + attr['value'] + '\n'

        # Drop name and values, groupby UID and sum grouped values,
        # ...reset index (ie. all attributes in single cell now, separated by
        # ...newlines as set up above)
        attr = attr.drop(['name', 'value'], axis=1).groupby('product_uid').sum().reset_index()

        # Create master data frame
        print("- Creating master data frame...")

        # Merge all data into one master dataframe by merging descriptions
        # ...and attributes onto training data on UID...
        # ...Fill any NAs with empty string
        data = pd.merge(train_df, prod_desc, how='left', on='product_uid').drop('id', axis=1).merge(attr, on='product_uid', how='left').fillna('')

        # Finally create a master index column, which will be used to reference
        # ...individual search terms
        data = data.reset_index().drop('index', axis=1).reset_index()

        return data

    def gen_features(self, preproc):
        # NLP
        print("- Applying spaCy NLP processor to all string features...")
        for feat in proc_feat:
            print('-- Applying to {}'.format(feat))
            data[feat] = data[feat].apply(nlp_tag)

        # Len of query
        print('- Creating length of query column...')
        data['q_len'] = data['search_term'].apply(lambda x: len(x))

        # Get common words between query and returned product title
        print('- Creating query-title common words column...')
        data['com_title'] = func_row(data[['search_term', 'product_title']], common_words_doc)

        # Get common words between query and returned product description
        print('- Creating query-description common words column...')
        data['com_desc'] = func_row(data[['search_term', 'product_description']], common_words_doc)

        # Get common words between query and returned product description
        print('- Creating query-attributes common words column...')
        data['com_attr'] = func_row(data[['search_term', 'attributes']], common_words_doc)

        # Clean up
        data = data.drop(proc_feat, axis=1)
