from sentence_transformers import SentenceTransformer, util
import torch
from collections import defaultdict
import re
encoder = SentenceTransformer('msmarco-distilbert-base-v2')

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

pattern1 = "<title>(.*?)\n<desc>"
pattern2 = "Description:\n(.*?)\n</top>"
pattern_id = "Number: (.*?)\n<title>"
with open('../resources/OHSUMED/t9.filtering/ohsu-trec/trec9-train/query.ohsu.1-63', 'r') as f:
    queries_ohsu = extract_queries(f)

queries_results_ohsu = defaultdict()
with open('../resources/OHSUMED/t9.filtering/ohsu-trec/trec9-train/qrels.ohsu.adapt.87', 'r') as f:
    raw = f.read().split('\n')[:-1]
    for r in raw:
        if queries_results_ohsu.get(r.split('\t')[0]) == None:
            queries_results_ohsu[r.split('\t')[0]] = [r.split('\t')[1]]
        else:
            queries_results_ohsu[r.split('\t')[0]].append(r.split('\t')[1])

indexes_used = [item for k,v in queries_results_ohsu.items() for item in v]



with open('../resources/ohsumed_corpora.txt', 'r') as f:
    ohsu = f.read()
texts = [text for i, text in enumerate(ohsu.split('\n')[:-1]) if i % 2 == 1]
ids = [id for i, id in enumerate(ohsu.split('\n')[:-1]) if i % 2 == 0]

texts_to_encode = []
not_in_dataset = []
ix2number = defaultdict()
for i in indexes_used:
    if i in ids:
        text_index = ids.index(i)
        ix2number[len(texts_to_encode)] = i
        texts_to_encode.append(texts[text_index])
    else:
        not_in_dataset.append(i)

print(len(texts_to_encode))



corpus_embeddings = encoder.encode(texts_to_encode, convert_to_tensor=True)

correct = 0
total = 0
for k,v in queries_results_ohsu.items():
    query_embedding = encoder.encode(queries_ohsu[k], convert_to_tensor=True)

    results = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    pred = []
    for i in results[0]:
        print(ix2number[i['corpus_id']], i['score'])
        pred.append(ix2number[i['corpus_id']])
    ground_truth = []
    for i in v:
        if i not in not_in_dataset:
            ground_truth.append(i)
    print(ground_truth)

    correct += len(set(pred).intersection(set(ground_truth)))
    total += len(ground_truth)

print(correct / total)
# print(ids.index('87070033'))
# print(len(not_in_dataset))