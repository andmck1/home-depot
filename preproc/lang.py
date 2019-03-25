# Author: Alastair Hamilton
# Date: April 2019
# Title: Model for Home-depot Kaggle Competition

# ---------------
# IMPORTS -------
# ---------------


# ---------------
# CONFIG --------
# ---------------


# ---------------
# FUNCTIONS -----
# ---------------

# Remove punctuation from column of dataframe
def rmv_punc(df, col):

    def punct_filter(doc):
        return tuple(filter(lambda y: not y.is_punct, doc))

    df.loc[:, col] = df.loc[:, col].apply(punct_filter)
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
            if w2.lemma_.find(w1.lemma_) >= 0:
                tot += 1
                break
    return tot
