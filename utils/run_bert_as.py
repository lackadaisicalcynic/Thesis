from bert_serving.client import BertClient
bc = BertClient()
f = open('/home/lackadacka/workspace/Thesis/resources/abstracts_sample.txt', 'r')
text = f.read()
absts = text.split('\n')[:-1]
vectors = bc.encode(absts)
with open('../resources/sample_vectors.txt', 'w')as f:
	for vec in vectors:
		for i in vec:	
			f.write('{} '.format(i))
		f.write('\n')
	

