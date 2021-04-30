text1 = 'Some patients converted from ventricular fibrillation to organized rhythms by defibrillation-trained ambulance technicians (EMT-Ds) will refibrillate before hospital arrival. The authors analyzed 271 cases of ventricular fibrillation managed by EMT-Ds working without paramedic back-up. Of 111 patients initially converted to organized rhythms, 19 (17%) refibrillated, 11 (58%) of whom were reconverted to perfusing rhythms, including nine of 11 (82%) who had spontaneous pulses prior to refibrillation. Among patients initially converted to organized rhythms, hospital admission rates were lower for patients who refibrillated than for patients who did not (53% versus 76%, P = NS), although discharge rates were virtually identical (37% and 35%, respectively). Scene-to-hospital transport times were not predictively associated with either the frequency of refibrillation or patient outcome. Defibrillation-trained EMTs can effectively manage refibrillation with additional shocks and are not at a significant disadvantage when paramedic back-up is not available.'
text2 = 'Of patients initially converted to organized rhythms, refibrillated, of whom were reconverted to perfusing rhythms, including nine of who had spontaneous pulses prior to refibrillation. Among patients initially converted to organized rhythms, hospital admission rates were lower for patients who refibrillated than for patients who did not (P = NS), although discharge rates were virtually identical (and , respectively). Scene-to-hospital transport times were not predictively associated with either the frequency of refibrillation or patient outcome. Defibrillation-trained EMTs can effectively manage refibrillation with additional shocks and are not at a significant disadvantage when paramedic back-up is not available.'
query1 = text1
text4 = 'Twenty one cases of amyloidosis of the lower respiratory tract were seen at a single center. In three patients, multifocal bronchial amyloid plaques led to stenosis and atelectasis, and in two, small pseudotumor masses were an incidental bronchoscopic finding. Two patients had nodular parenchymal amyloidosis, in one of whom the lesions were progressive and in the other static. Fifteen patients had diffuse parenchymal amyloidosis. Two of these had severe interstitial involvement and died in respiratory failure; eight had congestive cardiac failure, and parenchymal amyloidosis was a post-mortem finding; two had senile cardiorespiratory amyloidosis, also found at autopsy; and in three, the amyloidosis was associated with malignancy. The degree of respiratory embarrassment seemed to be related to the amount of amyloid in the gas diffusion zones, irrespective of the etiology of amyloidosis.'
text5 = 'Recurrent pulmonary embolism from the lower extremities or pelvis, despite anticoagulation, often requires interruption of the inferior vena cava (IVC). We report two patients in whom interruption of the IVC failed to ameliorate symptoms. Both patients demonstrated a previously unrecognized duplication of the IVC. We stress the importance of excluding abdominal venous anomalies prior to interrupting the IVC using surgical or percutaneous methods.'
query2 = 'IVC'

from bert_serving.client import BertClient
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer, CrossEncoder, util
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2")
#
# model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2")


encoder = SentenceTransformer('msmarco-distilbert-base-v2')
vectors = encoder.encode([text1, text2, text4, text5])
queries = encoder.encode([query1, query2])
# f = vectors.shape[1]
# t = AnnoyIndex(f, 'angular')
# for i, el in enumerate(vectors):
#     t.add_item(i, el)

# t.build(3)
# print(t.get_nns_by_vector(vectors[0], 5, include_distances=True))
print(util.semantic_search(queries[0], vectors, top_k=4))
print(util.semantic_search(queries[1], vectors, top_k=4))

reranker = CrossEncoder('cross-encoder/stsb-distilroberta-base')

scores = reranker.predict([[query1, text1],[query1, text2], [query1, text4], [query1, text5]])

print(scores)
# hits = sorted(scores, key=lambda x: x, reverse=True)
# for hit in hits:
#   print("\t{:.3f}".format(hit))
