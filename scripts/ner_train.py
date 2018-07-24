import pandas as pd
import spacy as sp
import random

# Data Directory, change as necessary
data_dir = './data/kaggle_test/'

# Load language model
print('\nLoading language model...')
nlp = sp.load(data_dir + '/textcat_model')
# nlp = sp.blank('en')

if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
else:
    ner = nlp.get_pipe('ner')

print('Done!')

# Load Data
print('\nLoading dataset...')
ner_train = pd.read_json(
    path_or_buf=(data_dir + 'ner_train.json'), orient='records').to_records()

# e = [[tuple(ent) for ent in ent_rec] for ent_rec in ner_train['entities']]
# ner_train = ner_train[:2]
TRAIN_DATA = list(
    zip(ner_train['content'], [{
        'entities': entities
    } for entities in [[tuple(ent) for ent in ent_rec]
                       for ent_rec in ner_train['entities']]]))
print(str(len(ner_train)) + ' records loaded.')

for text, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])
print('Done!')

# Train NER
print('\nTraining model...')
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(50):
        random.shuffle(TRAIN_DATA)
        # print(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            # print([sentence], {'entities': tuple(annotations)})
            nlp.update(
                [text],  # batch of texts
                [annotations],  # batch of annotations
                drop=0.5,  # dropout - make it harder to memorise data
                sgd=optimizer,  # callable to update weights
                losses=losses)
        print(losses)

# test the trained model
for text, annotations in TRAIN_DATA:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

# save model to output directory
nlp.to_disk(data_dir + '/textcat_ner_model')

# test the saved model

nlp2 = sp.load(data_dir + '/textcat_ner_model')
for text, _ in TRAIN_DATA:
    doc = nlp2(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])
