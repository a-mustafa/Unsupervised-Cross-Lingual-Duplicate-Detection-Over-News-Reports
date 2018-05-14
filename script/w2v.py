import os
import gensim
from gensim.models.doc2vec import LabeledSentence,TaggedDocument
from gensim.models import Doc2Vec
#from gensim.models import doc2vec
#, keyedvectors
# sentences='TextList.txt'
# sentences='D:/NLP/Mass dataset/LocalityDetectionDataset2/'+sentences
sentencesfile='../atrocitiesdata/TextList3.txt'
dataText=[]
f = open(sentencesfile, 'r')
for d in f:
  dataText.append((d.decode('utf-8')).split())
'''
model = gensim.models.Word2Vec(dataText, size=100, window=5, min_count=5, workers=4,iter=30)
fname='../atrocitiesdata/w2vmodel.model'
model.save(fname)
model = gensim.models.Word2Vec.load(fname)
'''
#print dataText[1:3]
#sentence = TaggedDocument(words=[u'some', u'words', u'here'], tags=[u'SENT_1'])

sentences = [TaggedDocument(words=words, tags=[i]) for i, words in enumerate(dataText)]
model = Doc2Vec(dataText, size=100, window=8, min_count=5, workers=4)
fname='../atrocitiesdata/d2vmodel.model'
model.save(fname)
model = Doc2Vec.load(fname)
print model['BBC'] 


#dataText_features = dataText_features.toarray()