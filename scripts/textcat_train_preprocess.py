# ---
# Script to preprocess text before modelling
# ---

# ---
# %%
# ---
# Set up environment variables
# ---

import pandas as pd
import re

# Data Directory, change as necessary
data_dir = './data/kaggle_test/'

# Categories to annotate
categories = ['na']

# Read in the input file
data = pd.merge(
        pd.read_csv(data_dir + 'annotations_tagged.csv'),
        pd.read_json(path_or_buf=data_dir + 'data_clean.json', orient='records'),
        how='left',
        on='post_id',
        validate='1:1')
data = data[categories + ['content']]

print('\nEnvironment variables set up successfully.')

# %%
# ---
# Preprocessing
# ---

# Initialise training data set
train = pd.DataFrame()
cats = {}

# Convert to JSON
for i in range(len(data)):
    if str(data.loc[i,'content']) == 'nan':
        continue   
    cats = (dict(zip(categories, data.loc[i, categories])))
    train = train.append({
        'content' : data.loc[i, 'content'],
        'cats' : cats
    }, ignore_index=True)

# Save training data to .JSON format
print('Done!')
train.to_json(path_or_buf=(data_dir + 'textcat_train.json'), orient='records')
print(train.info())
print(train.head(10))