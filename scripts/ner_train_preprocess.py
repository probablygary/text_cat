import spacy as sp
import pandas as pd

#
# ─── LOAD DATA AND VARIABLES ────────────────────────────────────────────────────
#

print('\nLoading Data...')
data_dir = './data/kaggle_test/'

# Read input file
tagged = pd.read_json(
    path_or_buf=data_dir + 'ner_tagged.json',
    orient='records').reset_index(drop=True)
data = pd.read_json(
    path_or_buf=data_dir + 'data_clean.json',
    orient='records').reset_index(drop=True)
print('Done!\n{} records loaded!'.format(len(tagged)))

#
# ─── PROCESS DATA ───────────────────────────────────────────────────────────────
#

print('\nProcessing Data...')
for ent_list in tagged['entities']:
    if len(ent_list[0]) < 1:
        del ent_list[0]
tagged = tagged[[len(i) > 0
                 for i in tagged['entities']]].reset_index(drop=True)

ner_train = pd.DataFrame()
entities = []
for i in range(len(tagged['entities'])):
    if i != 0 and tagged.loc[i, 'post_id'] != tagged.loc[i - 1, 'post_id']:
        ner_train = ner_train.append(
            pd.DataFrame({
                'post_id': tagged.loc[i - 1, 'post_id'],
                'entities': [entities]
            }))
        entities = []
    elif i == len(tagged['entities']) - 1:
        if tagged.loc[i, 'post_id'] != tagged.loc[i - 1, 'post_id']:
            ner_train = ner_train.append(
                pd.DataFrame({
                    'post_id': tagged.loc[i - 1, 'post_id'],
                    'entities': [entities]
                }))
            entities = []
            for ent in tagged.loc[i, 'entities']:
                entities.append(tuple(ent))
            ner_train = ner_train.append(
                pd.DataFrame({
                    'post_id': tagged.loc[i, 'post_id'],
                    'entities': [entities]
                }))
        else:
            for ent in tagged.loc[i, 'entities']:
                entities.append(tuple(ent))
                ner_train = ner_train.append(
                    pd.DataFrame({
                        'post_id': tagged.loc[i, 'post_id'],
                        'entities': [entities]
                    }))
    for ent in tagged.loc[i, 'entities']:
        entities.append(tuple(ent))

ner_train = pd.merge(
    ner_train,
    data,
    how='left',
    left_index=False,
    right_index=False,
    suffixes=('', ''),
    on='post_id')
ner_train.to_json(path_or_buf=data_dir + 'ner_train.json', orient='records')
print('Done!\n{} records in output.'.format(len(ner_train)))
print(ner_train.info())
print(ner_train.head(10))