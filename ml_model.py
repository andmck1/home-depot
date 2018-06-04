# Author: Alastair Hamilton
# Date: May/June 2018
# Title: ML Model for Home-depot Kaggle Competition


# Imports

## Data
import pandas as pd
import numpy as np

## Misc
import os
from pprint import pprint as pp

## Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

## ML
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Config

## Pandas error display OFF
pd.options.mode.chained_assignment = None

## Data Path
data_path = "./data/"

## Image Path
img_path = "./plots/"


# Get Data

## Get Features
features = pd.read_csv(data_path+'features.csv').drop(['Unnamed: 0','index','product_uid'], axis=1)


# Feature Analysis
print('Outputting feature analysis...')
      
## Create ratio features using query length
features['title_ratio'] = features['com_title'] / features['q_len']
features['desc_ratio'] = features['com_desc'] / features['q_len']
features['attr_ratio'] = features['com_attr'] / features['q_len']

## For each ratio, create a linear regression plot against relevance
for x in ['title_ratio', 'desc_ratio', 'attr_ratio']:
    bit = x.split('_')[0].title()
    ax = sns.lmplot(x='relevance', y=x, data=features, scatter_kws={'alpha':0.01})
    plt.title('Scatter Plot of Relevance Scores\nAgainst Ratio of Common Words in Query and {}'.format(bit))
    plt.xlabel('Relevance Scores')
    plt.ylabel('Query/{} Ratio'.format(bit))
    plt.savefig(img_path+'{}_ratio.png'.format(bit), dpi=500, bbox_inches="tight")


# ML Modelling
print('Starting ML Modelling...')

## Chose features
df = features.drop(['title_ratio', 'desc_ratio', 'attr_ratio'], axis=1)

## Split
X_train, X_test, y_train, y_test = train_test_split(df.drop('relevance',axis=1),
                                                    df['relevance'],
                                                    test_size=0.33, random_state=42)

## Gradient Boosted Tree Model
gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.4, loss='huber', alpha=0.9)

## Random forest Model
rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)

## For all models
for mdl in [rf, gbr]:
    
    ### Set Model Name
    if mdl is rf:
        model_name = 'Random_Forest'
    elif mdl is gbr:
        model_name = 'Gradient_Boosted_Tree'

    print('Running model: {}...'.format(model_name))
    
    ### Fit Model
    mdl.fit(X_train, y_train)
    print('Fitting Model...')
    
    ### Show variable importance
    print('Plotting Variable Importance...')
    importances = pd.np.nan
    if mdl is gbr or mdl is rf:
        importances = pd.Series(data=mdl.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    if importances is not pd.np.nan:
        ax = importances.plot(kind='bar')
        plt.title('Bar Plot of Relative Importance of Features using the {} Model'.format(model_name))
        plt.xlabel('Feature')
        plt.ylabel('Relative Importance')
        plt.savefig(img_path+'{}_relimp.png'.format(model_name), dpi=500, bbox_inches="tight")

    ### Predictions...
    print('Getting Predictions...')
    pred = mdl.predict(X_test)

    ### Metrics
    print('Calculating and Plotting Metrics...\n')
    print('MSE: {:.4f}\nMAE: {:.4f}\nR2: {:.4f}'.format(mean_squared_error(y_test, pred), mean_absolute_error(y_test, pred), r2_score(y_test, pred), mdl.score(X_test, y_test)))
    print('\nCross Validation')
    print(cross_val_score(mdl, X_train, y_train))

    ### Residuals
    s = np.std(pred)
    std_resids = (pred - y_test.values)/s
    df = pd.DataFrame({"std_resids":std_resids, "preds":pred})
    df['Range'] = df['std_resids'].apply(lambda x: 'r' if (x>2.0 or x<-2.0) else 'b')
    ax = df.plot(kind='scatter', x='preds', y='std_resids', alpha=0.01, c=df['Range'].values)
    plt.title('Scatter Plot of Residuals (in units of standard deviations)\nResidual Outwith 2 Stdev. has been Coloured Red')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual/Standard Deviation')
    print('Proportion of residuals outwith 2 standard deviations: {}'.format(df['Range'].value_counts()['b']/len(df)))
    plt.savefig(img_path+'{}_std_resids.png'.format(model_name), dpi=500, bbox_inches="tight")

    ### Deviance for GBT - testing for over-fitting
    if mdl is gbr:
        test_score = np.zeros((150,), dtype=np.float64)

        for i, y_pred in enumerate(mdl.staged_predict(X_test)):
            test_score[i] = mdl.loss_(y_test, y_pred)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Deviance')
        plt.plot(np.arange(150) + 1, mdl.train_score_, 'b-',
                 label='Training Set Deviance')
        plt.plot(np.arange(150) + 1, test_score, 'r-',
                 label='Test Set Deviance')
        plt.title('Line Plot of Deviance of Test and Train Predictions\nAgainst Boosting Iterations when using Gradient Boosted Tree')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
        plt.savefig(img_path+'gbr_dev.png', dpi=500, bbox_inches="tight")

