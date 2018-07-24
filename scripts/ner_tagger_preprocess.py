import pandas as pd
import os

#
# ─── ENVIRONMENT SETUP ──────────────────────────────────────────────────────────
#

# Data Directory, change as necessary
data_dir = './data/kaggle_test/'

# Read in data file
print('\nLoading dataset...')
ner_untagged_full = pd.DataFrame(
    pd.read_json(
        path_or_buf=(data_dir + 'ner_untagged_full.json'), orient='records'))
print('Done!')
#
# ─── PROCESS DATA ───────────────────────────────────────────────────────────────
#

# subset data, include only records tagged 'transport'
print('\nProcessing Data...')
textcat_tag = pd.read_csv(data_dir + 'annotations_tagged.csv', index_col=0)
textcat_tag = textcat_tag[textcat_tag['transport'] == 1].reset_index()
ner_untagged = pd.merge(
    ner_untagged_full,
    textcat_tag,
    how='right',
    left_index=False,
    right_index=False,
    suffixes=('', ''),
    on='post_id')
ner_untagged = ner_untagged[['post_id', 'sent_id', 'sentence', 'words']]
print('Done!')
#
# ─── OUTPUT DATA ────────────────────────────────────────────────────────────────
#

ner_untagged.to_json(
    path_or_buf=(data_dir + 'ner_untagged.json'), orient='records')
print(ner_untagged.head(10))
print(ner_untagged.info())