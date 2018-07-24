import pandas as pd
import re
import os

# ---
# Environment Setup
# ---

# Data Directory, change as necessary
data_dir = './data/kaggle_test/'

# Categories to annotate
categories = ['na']

untagged = pd.DataFrame()

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
    untagged = untagged.append(
        buf[buf['post_id'].isin(diff)], ignore_index=True)
    # drop_list = []
    # for i in range(len(untagged)):
    #     for category in categories:
    #         if untagged.loc[i, category] == 1 | 0:
    #             drop_list.append(i)
    #         else:
    #             continue
    # untagged = untagged.drop([i], axis=0)
else:
    untagged = untagged.append(
        pd.read_csv(data_dir + 'annotations_untagged.csv'), ignore_index=True)

for category in categories:
    if category not in untagged.columns.str.strip():
        untagged[category] = ''

print('{} record(s) loaded.'.format(str(len(untagged))))

# ---
# Annotation
# ---
tagged = pd.DataFrame()
i = 0
while i < len(untagged):
    df = pd.DataFrame()
    df = untagged.loc[i, ['post_id', 'title']]
    for category in categories:
        inp = ''
        if untagged.loc[i, category] not in [0, 1]:
            print('\n\nNO.:\t{}/{}'.format(len(tagged) + 1, len(untagged)))
            print('ID:\t{:<7}\nTITLE:\t{:<30}'.format(
                untagged.loc[i, 'post_id'], untagged.loc[i, 'title']))
            print('KW:\t{}\n'.format(untagged.loc[i, 'keywords']))
            print('CATEGORY:\t{}'.format(category))
            print('1: TRUE\n2: FALSE\n3: SKIP\n0: SAVE & QUIT')
            while inp not in [1, 2, 3, 0]:
                inp = int(input('-->'))
            if inp == 3:
                continue
            elif inp == 1:
                df[category] = 1
            elif inp == 2:
                df[category] = 0
            # elif inp == 9:
            #     break
            elif inp == 0:
                break
            # if len(tagged) < 1 or tagged[tagged['post_id'].isin(
            #     [df['post_id']])].empty:
            #     tagged = tagged.append(df)
            # else:
            #     tagged.loc[tagged['post_id'] == df['post_id'], category] = df[
            #         category]
        else:
            df[category] = untagged.loc[i, category]
    if inp == 0:
        break
    # elif inp == 9:
    #     i -= 1
    #     tagged = tagged.drop(labels=tagged.tail(1)['post_id'])
    else:
        tagged = tagged.append(df, ignore_index=True)
        i += 1
        if i % 5 == 0:
            tagged.to_csv(data_dir + 'annotations_tagged.csv')

tagged.to_csv(data_dir + 'annotations_tagged.csv')
print('Annotation Complete!')