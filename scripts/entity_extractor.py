import spacy as sp
import pandas as pd

# Data Directory, change as necessary
data_dir = './data/kaggle_test/'

# Load language model
print('\nLoading language model...')
nlp = sp.load(data_dir + '/textcat_model')

# Load Data
print('\nLoading dataset...')
data = pd.read_json(
    path_or_buf=data_dir + 'data_clean.json',
    orient='records').reset_index(drop=True)

data = data[40:50].reset_index()

for text in data['content']:
    doc = nlp(text)
    spans = list(doc.ents) + list(doc.noun_chunks)
    for span in spans:
        span.merge()
    for subject in doc:
        if subject.dep_ == 'nsubj' and subject.ent_type_ in [
                'ORG', 'PERSON', 'NORP'
        ] and subject.head.pos_ == 'VERB':
            print('AGENT: {}\nEVENT: {}\nOBJECT: {}\n'.format(
                subject.text.strip(), subject.head.text.strip(), [
                    t.text.strip() for t in subject.head.rights
                    if t.is_punct == False and t.is_stop == False
                ]))
