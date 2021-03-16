from bert_serving.client import BertClient
from elasticsearch import Elasticsearch
bc = BertClient()
hosts = ['localhost']
es = Elasticsearch(hosts)

query = bc.encode(['посттравматическое расстройство'])
source = {
  "script_score": {
    "query": {"match_all": {}},
    "script": {
      "source": "cosineSimilarity(params.query_vector, doc['article']) + 1.0",
      "params": {"query_vector": query[0]}
    }
  }
}
source2 = {
  "query": {
    "bool": {
      "must": {
        "match": {
          "article": 'patients'
        }
      }
    }
  }
}

# res = es.search(index="corpora", body=source2)
res = es.search(index="semantic_corpora_rubert", body={'size': 10,'query': source})
# res = es.search(index="parallel_corpora_rubert", body= source)
print("Got %d Hits:" % res['hits']['total']['value'])
for hit in res['hits']['hits']:
    print("%(timestamp)s %(article)s" % hit["_source"])
    print('id: ', hit["_id"])