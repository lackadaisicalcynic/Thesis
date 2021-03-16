from datetime import datetime
import glob
import numpy as np
from elasticsearch import Elasticsearch
hosts = ['localhost']
es = Elasticsearch(hosts)

files = glob.glob('../resources/ru-en/ru-en-release/*en.txt')
index_name = 'corpora'

# es.indices.delete(index='corpora')
# es.indices.delete(index='semantic_corpora')

# for i, file in enumerate(files):
#     with open(file, 'r') as f:
#         doc = {
#             'timestamp': datetime.utcnow(),
#             'article': f.read()
#         }
#         res = es.index(index=index_name, body=doc, id=i)
#
#
source = {
    "mappings": {
        "properties": {
            "quote": {"type": "text"},
            "vector": {"type": "dense_vector", "dims": 768}
        }
    }
}
# es.indices.create(index='semantic_corpora', body=source)
# with open('../resources/parallel_corpus_vectors.txt', 'r') as f:
#     embeddings = f.read()
# embeddings = np.array([emb.split(' ')[:-1] for emb in embeddings.split('\n')[:-1]], dtype=np.float32)
# for i, emb in enumerate(embeddings):
#     doc = {
#         'timestamp': datetime.utcnow(),
#         'article': emb
#     }
#     res = es.index(index='semantic_corpora', body=doc, id=i)
#
# es.indices.delete(index='semantic_corpora_biobert')
# es.indices.create(index='semantic_corpora_biobert', body=source)
# with open('../resources/parallel_corpus_vectors_biobert.txt', 'r') as f:
#     embeddings = f.read()
# embeddings = np.array([emb.split(' ')[:-1] for emb in embeddings.split('\n')[:-1]], dtype=np.float32)
# print(len(embeddings))
# for i, emb in enumerate(embeddings):
#     doc = {
#         'timestamp': datetime.utcnow(),
#         'article': emb
#     }
#     res = es.index(index='semantic_corpora_biobert', body=doc, id=i)


es.indices.delete(index='semantic_corpora_rubert')
es.indices.create(index='semantic_corpora_rubert', body=source)
with open('../resources/parallel_corpus_vectors_rubert.txt', 'r') as f:
    embeddings = f.read()
embeddings = np.array([emb.split(' ')[:-1] for emb in embeddings.split('\n')[:-1]], dtype=np.float32)
for i, emb in enumerate(embeddings):
    doc = {
        'timestamp': datetime.utcnow(),
        'article': emb
    }
    res = es.index(index='semantic_corpora_rubert', body=doc, id=i)