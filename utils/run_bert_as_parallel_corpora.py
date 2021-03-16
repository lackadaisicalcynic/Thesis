from bert_serving.client import BertClient
import glob
bc = BertClient()

files = glob.glob('../resources/ru-en/ru-en-release/*ru.txt')
index_name = 'corpora'

text = []
for file in files:
    with open(file, 'r') as f:
        text.append(f.read())
vectors = bc.encode(text)

with open('../resources/parallel_corpus_vectors_rubert.txt', 'w') as out:
    for vec in vectors:
        for i in vec:
            out.write('{} '.format(i))
        out.write('\n')