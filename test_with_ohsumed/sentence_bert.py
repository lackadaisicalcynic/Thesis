from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from collections import defaultdict
import re
encoder = SentenceTransformer('stsb-distilbert-base')
reranker = CrossEncoder('cross-encoder/ms-marco-electra-base')

PREDICTION_AT = 10


def extract_queries(f):
    pattern1 = "<title>(.*?)\n<desc>"
    pattern2 = "Description:\n(.*?)\n</top>"
    pattern_id = "Number: (.*?)\n<title>"

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

def parse_query_results():
    queries_results_ohsu = defaultdict()
    with open('../resources/OHSUMED/ohsu-trec/trec9-train/qrels.ohsu.batch.87', 'r') as f:
        raw = f.read().split('\n')[:-1]
        for r in raw:
            if r.split('\t')[2] == '2':
                if queries_results_ohsu.get(r.split('\t')[0]) == None:
                    queries_results_ohsu[r.split('\t')[0]] = [r.split('\t')[1]]
                else:
                    queries_results_ohsu[r.split('\t')[0]].append(r.split('\t')[1])

    return queries_results_ohsu

def build_corpora(ohsu):
    texts = [text for i, text in enumerate(ohsu.split('\n')[:-1]) if i % 2 == 1]
    ids = [id for i, id in enumerate(ohsu.split('\n')[:-1]) if i % 2 == 0]

    # new
    corpora_dict = defaultdict()
    for id, text in zip(ids, texts):
        corpora_dict[id] = text

    return corpora_dict

with open('../resources/OHSUMED/ohsu-trec/trec9-train/query.ohsu.1-63', 'r') as f:
    queries_ohsu = extract_queries(f)


queries_results_ohsu = parse_query_results()
indexes_used = [item for k,v in queries_results_ohsu.items() for item in v]



with open('../resources/ohsumed_corpora.txt', 'r') as f:
    ohsu = f.read()


corpora_dict = build_corpora(ohsu)


texts_to_encode = []
not_in_dataset = []
ix2ohsu_id = defaultdict()
for i in indexes_used:
    if i in corpora_dict.keys():
        ix2ohsu_id[len(texts_to_encode)] = i
        texts_to_encode.append(corpora_dict[i])
    else:
        not_in_dataset.append(i)

print(len(texts_to_encode))


corpus_embeddings = encoder.encode(texts_to_encode, convert_to_tensor=True)

correct = 0
total = 0
f = open('outputs.txt', 'w')

for k, label in queries_results_ohsu.items():
    query_embedding = encoder.encode(queries_ohsu[k], convert_to_tensor=True)

    ground_truth = []
    for i in label:
        if i not in not_in_dataset:
            ground_truth.append(i)

    results = util.semantic_search(query_embedding, corpus_embeddings, top_k=PREDICTION_AT*5)

    # new
    texts_for_reranking = defaultdict()

    pred = []
    for i in results[0]:
        # print(ix2ohsu_id[i['corpus_id']], i['score'])
        pred.append(ix2ohsu_id[i['corpus_id']])
        #new
        texts_for_reranking[ix2ohsu_id[i['corpus_id']]] = corpora_dict[ix2ohsu_id[i['corpus_id']]]

    #new
    if results[0] != []:
        model_inputs = [[queries_ohsu[k], passage] for _, passage in texts_for_reranking.items()]
        rerank_predictions = reranker.predict(model_inputs)
        pred_dict = defaultdict()
        for pred, key in zip(rerank_predictions, texts_for_reranking.keys()):
            pred_dict[key] = pred
        pred_dict = {k:v for k, v in sorted(pred_dict.items(), key=lambda x:x[1], reverse=True)}
        pred_dict = [(k,v) for k, v in pred_dict.items()]

        # for k, v in pred_dict[:len(ground_truth)]:
        #     print(f'{k}  {v}')
        pred = [i[0] for i in pred_dict[:PREDICTION_AT]]
        pred_scores = [i[1] for i in pred_dict[:PREDICTION_AT]]
        print(f'prediction: {pred}')
        print(f'scores: {pred_scores}')
        # something_valuable = False
        # for rr in rerank_predictions:
        #     if rr > 0.6:
        #         something_valuable = True
        #         result = rr

    ground_truth = []
    for i in label:
        if i not in not_in_dataset:
            ground_truth.append(i)
    print(f'ground truth: {ground_truth}')
    print()

    # if something_valuable:
    #     print('looks similar!\n', result)

    #filw writing
    f.write('Query:\n' + queries_ohsu[k])
    f.write('\n\nGround Truth:\n')
    for gt in ground_truth:
        f.write(corpora_dict[gt] + '\n')
    f.write('\nPredictions:\n')

    for p in pred:
        f.write(corpora_dict[p] + '\n')
    f.write('-'*200 +'\n')


    inter = set(pred).intersection(set(ground_truth))
    correct += len(inter)
    if len(inter) > 0:
        print(inter)
        print()
    # total += len(ground_truth)
    total += PREDICTION_AT

print(f'accuracy: {correct / total}')
# print(ids.index('87070033'))
# print(len(not_in_dataset))
f.close()