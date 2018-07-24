import pandas as pd
import re
from math import ceil
from time import asctime

#
# ─── LOAD VARIABLES AND FILES ───────────────────────────────────────────────────
#

# Data Directory, change as necessary
data_dir = './data/kaggle_test/'

# Read data file
print('Loading data...')
data = pd.read_csv(data_dir + 'data.csv', index_col=0)
print('{} records loaded!\n'.format(len(data)))

#
# ─── FUNCTIONS FOR CLEANING DATA ────────────────────────────────────────────────
#


def compress_whitespace(text):
    """Compress multi-line whitespace to single line"""

    whitespace = ('\r\n', '\r', '\n')
    for ws in whitespace:
        spaces = re.finditer('(' + ws + ')' + '+', text)
        for i in spaces:
            text = re.sub(re.escape(i.group()), ws, text)
    return text


def remove_noise(text):
    """Remove noise words like URLs, ellipsis, etc."""

    noise_words = ('\[...\]', '\[…\]', 'www\.', 'http://', 'https://',
                   '\.com\W', '\.org\W', '\.sg\W')
    for nw in noise_words:
        noise = re.finditer(nw, text)
        for i in noise:
            text = re.sub(re.escape(i.group().strip()), '', text)
    return text


def remove_sections(text):
    """Remove irrelevant sections of articles like related article links, captions for photos, etc."""

    sections = ('[\s*](Related article)[\s*\S*]*', '[\s*](Top photo)[\s*\S*]*',
                '[\s*](Top image)[\s*\S*]*',
                '[\s*](Vaguely related article)[\s*\S*]*')
    for sect in sections:
        sects = re.finditer(sect, text)
        for i in sects:
            text = re.sub(re.escape(i.group()), '', text)
    return text


#
# ─── CLEAN DATA AND OUTPUT ──────────────────────────────────────────────────────
#

print('Cleaning data...')
pct = list(range(0, 100, 5))
for i in range(len(data['content'])):
    if ceil(i*100/len(data['content'])) in pct:
        print('{:>5}% Done\t{}'.format(ceil(i*100/len(data['content'])), asctime()))
        pct.remove(ceil(i*100/len(data['content'])))
    text = data.loc[i, 'content']
    text = remove_noise(text)
    text = compress_whitespace(text)
    text = remove_sections(text)
    data.loc[i, 'content'] = text
data.to_json(path_or_buf=data_dir + 'data_clean.json', orient='records')
print('Done!\n')
print(data.head(10))
print(data.loc[len(data)-1, 'content'])