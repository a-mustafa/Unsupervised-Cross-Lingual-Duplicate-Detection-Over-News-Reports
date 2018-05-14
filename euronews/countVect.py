import json
from sklearn.cross_decomposition import CCA
import time
import sys
import random
from os import listdir
from os.path import isfile, join
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
import cPickle
from multiprocessing import Pool,cpu_count
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from random import shuffle
import re
import numpy as np


directorypath='/home/ahmad/duplicate-detection/euronews/data/jsonfiles2/'
jsondata = [join(directorypath, f) for f in listdir(directorypath) if isfile(join(directorypath, f))]
shuffle(jsondata)

stpwords=set(stopwords.words("spanish")+stopwords.words("english")+stopwords.words("german")+stopwords.words("french"))
alllngpairs=[['en','ar'],['en','fa'],['en','fr'],['en','es'],['en','de'],['ar','fa'],['ar','fr'],['ar','es'],['ar','de'],['fa','fr'],['fa','es'],['fa','de'],['fr','es'],['fr','de'],['es','de']]

print "EuroNews Supervised CCA (CountVectorizer) results..."

for lngp in alllngpairs[3:4]:
  eventsentspr=[]
  #filenm=jsondata[0]
  for idx,filenm in enumerate(jsondata):
    with open(filenm,"r") as myfile:
      dayjson=json.load(myfile)
    
    for event in dayjson:
      if any([ll not in event.keys() for ll in lngp]):
          continue
      
      eventsentspr.append([event[lngp[0]]['title'],event[lngp[1]]['title']])
      #lngpair.append(lngp)
  
  shuffle(eventsentspr)
  possentleft=[]
  possentright=[]
  
  for pr in eventsentspr:
    possentleft.append(pr[0])
    possentright.append(pr[1])
  
  npos=len(possentleft)
  
  negsentleft=[]
  negsentright=[]
  for _ in range(2*npos):
    negidx=list(np.random.choice(npos,2, replace=False))
    negsentleft.append(possentleft[negidx[0]])
    negsentright.append(possentright[negidx[1]])
  
  dataText_vectorizer1 = CountVectorizer(analyzer = "word", stop_words=stpwords, binary=True) #,ngram_range=(1, 1), max_features = 1000 
  dataText_features1 = dataText_vectorizer1.fit_transform(possentleft+negsentleft)
  dataText_features1 = dataText_features1.toarray()
  
  dataText_vectorizer2 = CountVectorizer(analyzer = "word", stop_words=stpwords, binary=True) #,ngram_range=(1, 1), max_features = 1000 
  dataText_features2 = dataText_vectorizer2.fit_transform(possentright+negsentright)
  dataText_features2 = dataText_features2.toarray()
  
  #cca = CCA(n_components=min(dataText_features1.shape[1],dataText_features2.shape[1])/5)
  cca = CCA(n_components=min(10,min(dataText_features1.shape[1],dataText_features2.shape[1])/3))
  #Training with positive points
n_tr=npos/3
train1=np.array(dataText_features1[:n_tr])
train2=np.array(dataText_features2[:n_tr])
cca.fit(train1, train2)
timenow=time.time()
X_l, X_r = cca.transform(dataText_features1[n_tr,].reshape(1,-1), dataText_features2[n_tr,].reshape(1,-1))
a=cosine_similarity(X_l.reshape(1,-1),X_r.reshape(1,-1))
elapsedtime=time.time()-timenow
  #Testing positive points
  X_l, X_r = cca.transform(dataText_features1[n_tr:npos,], dataText_features2[n_tr:npos,])
  sim=[]
  for _X_l, _X_r in zip(X_l, X_r):
    sim.append(cosine_similarity(_X_l.reshape(1,-1),_X_r.reshape(1,-1)))
  
  TP=sum([True for _s in sim if _s >=0.5])
  FN=sum([True for _s in sim if _s < 0.5])
  
  #Testing negative points
  X_l, X_r = cca.transform(dataText_features1[npos:], dataText_features2[npos:])
  sim=[]
  for _X_l, _X_r in zip(X_l, X_r):
    sim.append(cosine_similarity(_X_l.reshape(1,-1),_X_r.reshape(1,-1)))
  
  FP=sum([True for _s in sim if _s >=0.5])
  TN=sum([True for _s in sim if _s < 0.5])
  
  
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
  F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
  
  print lngp[0],lngp[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))


print "EuroNews unsupervised CCA (CountVectorizer) results..."


for lngp in alllngpairs[3:4]:
  eventsentspr=[]
  #filenm=jsondata[0]
  for idx,filenm in enumerate(jsondata):
    with open(filenm,"r") as myfile:
      dayjson=json.load(myfile)
    
    for event in dayjson:
      if any([ll not in event.keys() for ll in lngp]):
          continue
      
      eventsentspr.append([event[lngp[0]]['title'],event[lngp[1]]['title']])
      #lngpair.append(lngp)
  
  shuffle(eventsentspr)
  possentleft=[]
  possentright=[]
  
  for pr in eventsentspr:
    possentleft.append(pr[0])
    possentright.append(pr[1])
  
  npos=len(possentleft)
  
  negsentleft=[]
  negsentright=[]
  for _ in range(2*npos):
    negidx=list(np.random.choice(npos,2, replace=False))
    negsentleft.append(possentleft[negidx[0]])
    negsentright.append(possentright[negidx[1]])
  
  dataText_vectorizer1 = CountVectorizer(analyzer = "word", stop_words=stpwords, binary=True) #,ngram_range=(1, 1), max_features = 1000
  dataText_features1 = dataText_vectorizer1.fit_transform(possentleft+negsentleft)
  dataText_features1 = dataText_features1.toarray()
  
  dataText_vectorizer2 = CountVectorizer(analyzer = "word", stop_words=stpwords, binary=True) #,ngram_range=(1, 1), max_features = 1000
  dataText_features2 = dataText_vectorizer2.fit_transform(possentright+negsentright)
  dataText_features2 = dataText_features2.toarray()
  
  
  cca = CCA(n_components=min(10,min(dataText_features1.shape[1],dataText_features2.shape[1])/3))
  X_l, X_r = cca.fit_transform(dataText_features1[:-1,], dataText_features2[:-1,])
  sim=[]
  for _X_l, _X_r in zip(X_l, X_r):
    sim.append(cosine_similarity(_X_l.reshape(1,-1),_X_r.reshape(1,-1)))
  
  FP=sum([True for _s in sim[npos:] if _s >=0.5])
  TN=sum([True for _s in sim[npos:] if _s <0.5])
  TP=sum([True for _s in sim[:npos] if _s >=0.5])
  FN=sum([True for _s in sim[:npos] if _s <0.5])
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
  F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
  
  print lngp[0],lngp[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))



