import re
import numpy as np
from annoy import AnnoyIndex
from collections import defaultdict
from bert_serving.client import BertClient


def extract_queries(f):
    queries = dict()
    text = f.read()
    query1 = []
    query2 = []
    qids = []
    for m in re.finditer(pattern1, text):
        query1.append(m.group(1))
    for m in re.finditer(pattern2, text):
        query2.append(m.group(1))
    for m in re.finditer(pattern_id, text):
        qids.append(m.group(1))
    for q1, q2, qid in zip(query1, query2, qids):
        queries[qid] = q2
    return queries


with open('../resources/ohsumed_vectors.txt', 'r') as f:
    vectors = f.read()
vectors = np.array([vector.split(' ')[:-1] for vector in vectors.split('\n')[:-1]], dtype=np.float32)

with open('../resources/ohsumed_corpora.txt', 'r') as f:
    ohsu = f.read()
indices = [id for i, id in enumerate(ohsu.split('\n')[:-1]) if i % 2 == 0]

# queries = dict()
pattern1 = "<title>(.*?)\n<desc>"
pattern2 = "Description:\n(.*?)\n</top>"
pattern_id = "Number: (.*?)\n<title>"
with open('../resources/OHSUMED/t9.filtering/ohsu-trec/trec9-train/query.ohsu.1-63', 'r') as f:
    queries_ohsu = extract_queries(f)

with open('../resources/OHSUMED/t9.filtering/ohsu-trec/trec9-train/query.mesh.1-4904', 'r') as f:
    queries_msh = extract_queries(f)

queries_results_ohsu = defaultdict()
with open('../resources/OHSUMED/t9.filtering/ohsu-trec/trec9-train/qrels.ohsu.adapt.87', 'r') as f:
    raw = f.read().split('\n')[:-1]
    for r in raw:
        if queries_results_ohsu.get(r.split('\t')[0]) == None:
            queries_results_ohsu[r.split('\t')[0]] = [r.split('\t')[1]]
        else:
            queries_results_ohsu[r.split('\t')[0]].append(r.split('\t')[1])

queries_results_msh = defaultdict()
with open('../resources/OHSUMED/t9.filtering/ohsu-trec/trec9-train/qrels.mesh.adapt.87', 'r') as f:
    raw = f.read().split('\n')[:-1]
    for r in raw:
        if queries_results_msh.get(r.split('\t')[0]) == None:
            queries_results_msh[r.split('\t')[0]] = [r.split('\t')[1]]
        else:
            queries_results_msh[r.split('\t')[0]].append(r.split('\t')[1])


bc = BertClient()
f = vectors.shape[1]
annoy2id = defaultdict()
t = AnnoyIndex(f, 'angular')
for i, el in enumerate(zip(indices, vectors)):
    id, vec = el
    annoy2id[i] = id
    t.add_item(i, vec)

t.build(1000)
correct = 0
total = 0
missed_texts = 0
for k,v in queries_results_ohsu.items():

    # print(t.get_nns_by_vector(bc.encode([queries_ohsu[k]])[0], len(v), include_distances=True))
    results = t.get_nns_by_vector(bc.encode([queries_ohsu[k]])[0], 50, include_distances=True)
    for a, b in zip(results[0], results[1]):
        print(a, b, end=' ')
    print()
    results = [annoy2id[r[0]] for r in zip(results[0], results[1])]
    correct += len(set(results).intersection(set(v)))
    total += len(v)
    print(results)
    print(v)
    print()

    for id in v:
        if id not in indices:
            missed_texts += 1

print(correct / total)
print(missed_texts)

# correct = 0
# total = 0
# for k,v in queries_results_msh.items():
#
#     # print(t.get_nns_by_vector(bc.encode([queries_ohsu[k]])[0], len(v), include_distances=True))
#     results = t.get_nns_by_vector(bc.encode([queries_msh[k]])[0], len(v))
#     results = [annoy2id[r] for r in results]
#     correct += len(set(results).intersection(set(v)))
#     total += len(v)
#
#     for id in v:
#         if id not in indices:
#             missed_texts += 1
#
#
# print(correct / total)


# for i in range(vectors.shape[0]):
#     dist.append(t.get_nns_by_item(i, 2, include_distances=True))

# dist = [([i, item[0][1]], item[1][1]) for i, item in enumerate(dist)]

#
# for i, id in enumerate(ids):
#     check = re.sub('[0-9]', '', id)
#     if check != '':
#         print(i)

# print(missed_texts)