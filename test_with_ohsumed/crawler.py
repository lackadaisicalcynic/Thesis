from bert_serving.client import BertClient


f1 = open('../resources/OHSUMED/t9.filtering/ohsu-trec/trec9-train/ohsumed.87', 'r')

s = f1.read()
texts = s.split('.I ')

texts = [t.split('.W\n')[1] for t in texts if not len(t.split('.W\n')) < 2]
texts = [i.split('.A\n')[0] for i in texts]

ids = s.split('.U\n')[1:]
ids = [i.split('.S\n')[0] for i in ids]

f2 = open('../resources/ohsumed_corpora.txt', 'w')
for text, id in zip(texts, ids):
    f2.write(id.strip() + '\n')
    f2.write(text.strip() + '\n')

with open('../resources/ohsumed_corpora.txt', 'r') as f:
    ohsu = f.read()
texts = [text for i, text in enumerate(ohsu.split('\n')[:-1]) if i % 2 == 1]
ids = [id for i, id in enumerate(ohsu.split('\n')[:-1]) if i % 2 == 0]

bc = BertClient()
vectors = bc.encode(texts)
with open('../resources/ohsumed-vectors.txt', 'w') as f:
    for vec in vectors:
        for i in vec:
            f.write('{} '.format(i))
        f.write('\n')
