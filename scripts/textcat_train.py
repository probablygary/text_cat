import pandas as pd
import spacy as sp
import random
from time import asctime
import os

#
# ─── SET UP ENVIRONMENT VARIABLES ───────────────────────────────────────────────
#

# Data Directory, change as necessary
data_dir = './data/kaggle_test/'

# Categories to predict
categories = ['na']

# Load language model
print('\nLoading language model...')
nlp = sp.load('en_core_web_md')
# nlp = sp.blank('en')
# nlp = sp.load(data_dir + '/textcat_model')
if 'textcat' not in nlp.pipe_names:
    textcat = nlp.create_pipe('textcat')
    nlp.add_pipe(textcat, last=True)
else:
    textcat = nlp.get_pipe('textcat')

for category in categories:
    textcat.add_label(category)

print('Done!')

# Read in the input file
print('\nLoading dataset...')
data = pd.read_json(
    path_or_buf=(data_dir + 'textcat_train.json'),
    orient='records').to_records()
print(str(len(data)) + ' records loaded.')

#
# ─── PREPROCESSING ──────────────────────────────────────────────────────────────
#

print('\nProcessing data...')


def load_data(data, limit=0, split=0.8):
    random.shuffle(data)
    data = data[-limit:]
    # _, _, texts, cats = zip(*data)
    # cats = [{'toxic': 1 if x >= 1 else 0} for x in labels]
    texts = data['content']
    cats = data['cats']
    split = int(len(data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    return {
        'textcat_p': precision,
        'textcat_r': recall,
        'textcat_f': f_score,
        'textcat_a': accuracy
    }


(train_texts, train_cats), (val_texts, val_cats) = load_data(
    data=data, limit=len(data), split=0.8)
train = list(zip(train_texts, [{'cats': cats} for cats in train_cats]))
print('Done!')

#
# ─── TRAIN MODEL ────────────────────────────────────────────────────────────────
#

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
with nlp.disable_pipes(*other_pipes):  # only train textcat
    optimizer = nlp.begin_training()
    print('Training Model...')
    print('{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}'.format(
        '', 'LOSS', 'P', 'R', 'F', 'A'))
    for i in range(20):
        losses = {}
        batches = sp.util.minibatch(
            train, size=sp.util.compounding(4., 32., 1.001))
        for batch in batches:
            text, annotations = zip(*batch)
            nlp.update(
                text, annotations, sgd=optimizer, drop=0.2, losses=losses)
        with textcat.model.use_params(optimizer.averages):
            scores = evaluate(nlp.tokenizer, textcat, val_texts, val_cats)
        print(
            '{:>}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'  # print a simple table
            .format(i + 1, losses['textcat'], scores['textcat_p'],
                    scores['textcat_r'], scores['textcat_f'],
                    scores['textcat_a']))
print('Done!')
nlp.to_disk(data_dir + '/textcat_model')

# Save to log file
desc = 'Training with cleaned data containing 3646 records'


def check_dir(path):
    flag = False
    with os.scandir(path) as path:
        for entry in path:
            if os.DirEntry.is_file(entry) and entry.name == 'textcat_log.txt':
                flag = True
    path.close()
    return flag


if check_dir(data_dir):
    with open((data_dir + 'textcat_log.txt'), 'a') as f:
        print(
            '{:<}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:<}\n'.format(
                asctime(), losses['textcat'], scores['textcat_p'],
                scores['textcat_r'], scores['textcat_f'], scores['textcat_a'],
                desc),
            file=f)
else:
    with open((data_dir + 'textcat_log.txt'), 'x') as f:
        print(
            '{:^24}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}'.format(
                '', 'LOSS', 'P', 'R', 'F', 'A', 'Description'),
            file=f)
        print(
            '{:<}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:<}\n'.format(
                asctime(), losses['textcat'], scores['textcat_p'],
                scores['textcat_r'], scores['textcat_f'], scores['textcat_a'],
                desc),
            file=f)

#
# ─── TEST THE MODEL ─────────────────────────────────────────────────────────────
#

test1 = """Tesla today announced that it's fleet of electric vehicles will be unveiled on Singapore roads."""
test2 = """"I'm now talking about housing and things similar to houses and living in a 4-room flat and stuff; maybe housing development and stuff"""

doc1 = nlp(test1)
doc2 = nlp(test2)

print(test1, doc1.cats)
print(test2, doc2.cats)

print('Done!')