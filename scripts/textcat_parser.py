# %%
import spacy as sp
import pandas as pd
import re
import collections

# ---
# Environment Setup
# ---

# Load language model
print('Loading language model...')
nlp = sp.load('en_core_web_md')
print('Done!')

# %%
# Data Directory, change as necessary
data_dir = './data/kaggle_test/'

# Read in data file
print('Loading dataset...')
data = pd.read_csv(data_dir + 'data.csv').to_records()
# data = data[:10]
print('{} record(s) loaded.'.format(str(len(data))))

# ---
# Preprocessing
# ---
# %%
print('Processing data...')

# Load data into DF
_, _, texts, post_ids, titles = zip(*data)
keywords = []
untagged = pd.DataFrame()

for text in texts:
    print(texts.index(text))
    doc = nlp(str(text))
    nouns = [token.lemma_ for token in doc if re.match('^NN.', token.tag_) and not (token.is_stop or token.is_punct or token.is_digit)]
    if len(nouns)>0:
        top_n, _ = zip(*collections.Counter(nouns).most_common(10))
    else:
        top_n = 'No keywords'
    top_n = list(top_n)
    words = ''
    while len(top_n) > 1:
        words = words + top_n.pop() + ', '
    words = words + top_n.pop()
    keywords.append(words)

untagged = pd.DataFrame({
    'post_id': post_ids,
    'title': titles,
    'keywords': keywords,
})

# categories = ['hdb', 'pap']
# for category in categories:
#     untagged[category] = ''

untagged.to_csv(data_dir + 'annotations_untagged.csv')
print('Done!')
untagged.info()
untagged.head(5)