# ---
# Script for building a Pandas DB of posts extracted from JSON files
# ---

# %%
import pandas as pd
import glob
import os
import re

# Data Directory, change as necessary
data_dir = './data/kaggle_test/'

# Initialise pandas dataframe for input
data = pd.DataFrame()
results = pd.DataFrame()

# Search through folders, extract title and content from main articles
print('Building DB...')
for root, subFolders, files in os.walk(data_dir):
    for folder in subFolders:
        if re.match('^post.', folder):
            post_id = re.search('(?<=kaggle)([0-9]+)(?=_)',
                                folder).group(1)
            for f in glob.glob(
                    data_dir + folder + '/*Post*' + post_id + '*.json'):
                buffer = pd.read_json(
                    path_or_buf=('file://localhost/' + f),
                    typ='series',
                    encoding='utf-8')
                if buffer['content'].astype(str) == 'nan':
                    continue   
                df = pd.DataFrame(
                    {
                        'post_id': post_id,
                        'title': buffer['postTitle'],
                        'content': buffer['content'].strip()
                    },
                    index=[1])
                data = data.append(df, ignore_index=True)

# Save data to .csv format
data.to_csv(data_dir + 'data.csv')

print('Done!')
print('DB of {} records loaded.'.format(str(len(data))))
print(data.info())
print(data.head(5))