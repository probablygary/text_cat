import pandas as pd
import os
from math import ceil

#
# ─── ENVIRONMENT SETUP ──────────────────────────────────────────────────────────
#

# Data Directory, change as necessary
data_dir = './data/kaggle_test/'

# Read in data file
print('Loading dataset...')

ner_untagged = pd.DataFrame()
ner_tagged = pd.DataFrame()


def check_dir(path, file):
    flag = False
    with os.scandir(path) as path:
        for entry in path:
            if os.DirEntry.is_file(entry) and entry.name == str(file):
                flag = True
    path.close()
    return flag


#
# ─── PREPROCESSING ──────────────────────────────────────────────────────────────
#


def ner_tag(untagged):
    print('Processing data...')
    ner_cats = {
        'PERSON': 'People, including fictional.',
        'NORP': 'Nationalities or religious or political groups.',
        'FACILITY': 'Buildings, airports, highways, bridges, etc.',
        'ORG': 'Companies, agencies, institutions, etc.',
        'GPE': 'Countries, cities, states.',
        'LOC': 'Non-GPE locations, mountain ranges, bodies of water.',
        'PRODUCT': 'Objects, vehicles, foods, etc. (Not services.)',
        'EVENT': 'Named hurricanes, battles, wars, sports events, etc.',
        'WORK_OF_ART': 'Titles of books, songs, etc.',
        'LAW': 'Named documents made into laws.',
        'LANGUAGE': 'Any named language.',
        'DATE': 'Absolute or relative dates or periods.',
        'TIME': 'Times smaller than a day.',
        'PERCENT': 'Percentage, including "%".',
        'MONEY': 'Monetary values, including unit.',
        'QUANTITY': 'Measurements, as of weight or distance.',
        'ORDINAL': '"first", "second", etc.',
        'CARDINAL': 'Numerals that do not fall under another type.'
    }
    options = {'d': 'define', 'n': 'next sentence', 'q': 'save and quit'}
    tagged = pd.DataFrame()

    for i, word_list in enumerate(untagged['words']):
        inp = ''
        entities = [[]]

        # Print UI
        print('\n\n\n{}\n{}\t{:>5}/{:<5}\t{:>4}% Done\n\n{}\n'.format(
            '-' * 50, untagged.loc[i, 'post_id'],
            len(ner_tagged) + len(tagged),
            len(ner_untagged) + len(ner_tagged),
            ceil((len(ner_tagged) + len(tagged)) * 100 /
                 (len(ner_untagged) + len(ner_tagged))),
            untagged.loc[i, 'sentence']))
        for token in word_list:
            print('{:<15}\tstart: {:>3}\tend: {:>3}\ttag: {}'.format(
                token['token'], token['start'], token['end'], token['tag']))
        print('\n')
        for j, ent in enumerate(ner_cats):
            print('{:<2}\t{:10}\t{}'.format(j, ent, ner_cats[ent]))
        for option in options:
            print('{:<2}\t{}'.format(option, options[option]))

        # User Input
        while inp not in options:
            inp = input('-->')
        while inp != 'n':
            if inp == 'd':
                print('Define Named Entity:')
                start = input('start: ')
                # while start not in [
                #         str(k)
                #         for k in range(0, len(untagged.loc[i, 'sentence']))
                # ]:
                #     start = input('start: ')
                start = int(start)
                end = input('end: ')
                # while end not in [
                #         str(k)
                #         for k in range(0, len(untagged.loc[i, 'sentence']))
                # ] or int(end) < int(start):
                # end = input('end: ')
                while int(end) < int(start):
                    end = input('end: ')
                tag = input('tag: ')
                while tag not in [str(num) for num in range(0, len(ner_cats))]:
                    tag = input('tag: ')
                entities += [[int(start), int(end), list(ner_cats)[int(tag)]]]
            elif inp == 'q':
                break
            inp = input('-->')
        if inp == 'q':
            break

        tagged = tagged.append(
            {
                'post_id': untagged.loc[i, 'post_id'],
                'sent_id': untagged.loc[i, 'sent_id'],
                'sentence': untagged.loc[i, 'sentence'],
                'entities': entities
            },
            ignore_index=True)
        tagged.to_json(
            path_or_buf=data_dir + 'ner_tagged_TEMP.json', orient='records')
    return tagged


if check_dir(data_dir, 'ner_tagged.json'):
    ner_tagged = pd.read_json(
        path_or_buf=(data_dir + 'ner_tagged.json'), orient='records')
    ner_untagged = pd.read_json(
        path_or_buf=(data_dir + 'ner_untagged.json'), orient='records')
    untagged_ids = pd.Index(ner_untagged['sent_id']).difference(
        pd.Index(ner_tagged['sent_id']))
    print('{} record(s) loaded.'.format(str(len(ner_untagged))))
    ner_untagged = ner_untagged[ner_untagged['sent_id'].isin(
        untagged_ids)].reset_index()
    ner_tagged = ner_tagged.append(ner_tag(ner_untagged), ignore_index=True)
    ner_tagged.to_json(path_or_buf=(data_dir + 'ner_tagged.json'))
    print('Tagging complete!')

else:
    ner_untagged = pd.read_json(
        path_or_buf=(data_dir + 'ner_untagged.json'), orient='records')
    print('{} record(s) loaded.'.format(str(len(ner_untagged))))
    ner_tagged = ner_tagged.append(ner_tag(ner_untagged), ignore_index=True)
    ner_tagged.to_json(path_or_buf=(data_dir + 'ner_tagged.json'))
    print('Tagging complete!')
