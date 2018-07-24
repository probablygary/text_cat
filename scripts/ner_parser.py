import spacy as sp
import pandas as pd
import re
import collections
from math import ceil
from time import asctime

#
# ─── ENVIRONMENT SETUP ──────────────────────────────────────────────────────────
#

# Load language model
print('\nLoading language model...')
nlp = sp.load('en_core_web_md')
print('Done!')

# Data Directory, change as necessary
data_dir = './data/kaggle_test/'

# Read in data file
print('\nLoading dataset...')
data = pd.read_json(path_or_buf=data_dir + 'data_clean.json', orient='records')
# data = data[:10]
print('{} record(s) loaded.'.format(len(data)))

#
# ─── PREPROCESSING ──────────────────────────────────────────────────────────────
#

print('\nProcessing data...')
untagged = pd.DataFrame()
pct = list(range(0, 100, 5))
for i, post in enumerate(data['content']):
    if ceil(i * 100 / len(data['content'])) in pct:
        print('{:>5}% Done\t{}'.format(
            ceil(i * 100 / len(data['content'])), asctime()))
        pct.remove(ceil(i * 100 / len(data['content'])))

    doc = nlp(post)
    for sent in doc.sents:
        sent_doc = sent.as_doc()
        words = [{
            'token': token.text,
            'start': token.idx,
            'end': token.idx + len(token.text),
            'tag': token.ent_type_
        } for token in sent_doc if token.pos_ in ('NOUN', 'PROPN')]
        if len(words) < 1:
            continue
        untagged = untagged.append(
            {
                'post_id': data.loc[i, 'post_id'],
                'sentence': sent.text.strip(),
                'words': words
            },
            ignore_index=True)
untagged['sent_id'] = pd.Series(list(range(0, len(untagged['sentence']))))

#
# ─── OUTPUT DATA ────────────────────────────────────────────────────────────────
#

untagged.to_json(
    path_or_buf=(data_dir + 'ner_untagged_full.json'), orient='records')
print('Done!')
untagged.info()
print(untagged.head(5))