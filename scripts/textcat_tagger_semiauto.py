# %%
import pandas as pd
import re
import os
import spacy as sp

# ---
# Environment Setup
# ---
# %%
# Data Directory, change as necessary
data_dir = './data/kaggle_test/'

# Load language model
print('Loading language model...')
nlp = sp.load(data_dir + '/textcat_model')
textcat = nlp.get_pipe('textcat')

print('Done!')

# Categories to annotate
categories = ['na']
untagged = pd.DataFrame()

# Load Data set
print('Loading dataset...')


def check_dir(path):
    flag = False
    with os.scandir(path) as path:
        for entry in path:
            if os.DirEntry.is_file(
                    entry) and entry.name == 'annotations_tagged.csv':
                flag = True
    path.close()
    return flag


if check_dir(data_dir):
    untagged = untagged.append(
        pd.read_csv(data_dir + 'annotations_tagged.csv'))
    buf = pd.read_csv(data_dir + 'annotations_untagged.csv')
    diff = pd.Index(buf['post_id']).difference(pd.Index(untagged['post_id']))
    untagged = pd.merge(
        untagged,
        buf,
        how='left',
        left_index=False,
        right_index=False,
        validate='one_to_one',
        suffixes=('', ''))
    # untagged['checked'] = 1
    buf = buf[buf['post_id'].isin(diff)]
    # buf['checked'] = 0
    untagged = untagged.append(buf, ignore_index=True)

else:
    untagged = untagged.append(
        pd.read_csv(data_dir + 'annotations_untagged.csv'), ignore_index=True)
    # untagged['checked'] = 0

for category in categories:
    if category not in untagged.columns.str.strip():
        untagged[category] = ''

untagged = pd.merge(
    untagged,
    pd.read_csv(data_dir + 'data.csv')[['post_id', 'content']],
    how='left',
    on='post_id',
    validate='1:1')

print('{} record(s) loaded.'.format(str(len(untagged))))


# ---
# Annotation
# ---
def auto_annotate(batchsize=50):
    tagged = pd.DataFrame()
    i = 0
    while i < len(untagged):
        inp = ''
        j = 0
        buf = pd.DataFrame()
        while j < batchsize and i < len(untagged):
            flg = False
            df = pd.DataFrame()
            df = untagged.loc[i, ['post_id', 'title', 'keywords']]
            for category in categories:
                inp = ''
                if untagged.loc[i, category] not in [0, 1]:
                    flg = True
                    doc = nlp(str(untagged.loc[i, 'content']))
                    scores = textcat.predict([doc])
                    textcat.set_annotations([doc], scores[0])
                    for label, score in doc.cats.items():
                        df[str(label + '_score')] = score
                        if score > 0.5:
                            df[label] = 1
                        else:
                            df[label] = 0
            if flg:
                buf = buf.append(df, ignore_index=True)
                j += 1
            else:
                tagged = tagged.append(untagged.loc[i,:], ignore_index=True)

            i += 1
        for category in categories:
            buf = buf.sort_values(
                by=str(category + '_score'), ascending=False).reset_index()
            for k in range(len(buf)):
                inp = ''
                print('\n\n\nTOTAL:\t{}/{}'.format(i - len(buf) + k + 1, str(len(untagged))))
                print('BATCH:\t{}/{}'.format(k + 1, len(buf)))
                print('ID:\t{:<7}\nTITLE:\t{:<30}'.format(
                    buf.loc[k, 'post_id'], buf.loc[k, 'title']))
                print('KW:\t{}\n'.format(buf.loc[k, 'keywords']))
                print('CATEGORY: {}\t LABEL: {:<1}\t SCORE: {:<.3f}\t'.format(
                    category, buf.loc[k, category],
                    buf.loc[k, str(category + '_score')]))
                print('1: TRUE\n2: FALSE\n3: NEXT BATCH\n0: SAVE & QUIT')
                while inp not in [1, 2, 3, 0]:
                    inp = int(input('-->'))
                if inp == 1:
                    buf.loc[k, category] = 1
                elif inp == 2:
                    buf.loc[k, category] = 0
                elif inp in [0, 3]:
                    break
            if inp in [0, 3]:
                break
        tagged = tagged.append(buf, ignore_index=True)
        keep_cols = ['post_id', 'title'] + categories
        tagged = tagged[keep_cols]
        tagged.to_csv(data_dir + 'annotations_tagged.csv')
        if inp == 0:
            break

auto_annotate(50)
print('Annotation Complete!')