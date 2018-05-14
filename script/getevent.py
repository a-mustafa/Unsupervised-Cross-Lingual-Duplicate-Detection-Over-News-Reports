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
import urllib, json, urllib2
from contextlib import closing
import sklearn.metrics.pairwise.euclidean_distances
import sklearn

def loadfasttextmodel(filename):
  filename='/home/ahmad/fastText_multilingual/'
  w2v=dict()
  #['en','es','zh','hr','de','fa','ar','fr']['es','en','de']
  for lng in ['en','es','ar']:
    w2v[lng] = FastVector(vector_file=filename+'wiki.'+lng+'.vec')
    w2v[lng].apply_transform(filename+'alignment_matrices/'+lng+'.txt')
  
  return w2v


embeddingsmodel=loadfasttextmodel('Path To Vectors')


from nltk.corpus import stopwords
stopwords_list = set(stopwords.words("/home/ahmad/duplicate-detection/arabicstopwords.txt"))

stpwords=set(stopwords.words("spanish")+stopwords.words("english")+stopwords.words("german")+stopwords.words("french"))
stpwords=set(stopwords.words("spanish")+stopwords.words("english"))

def getembeddings(text,lng):
  wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', text)
  docembeddings=[]
  words=[]
  for word in wordslist:
    if '' !=word.strip().lower() and word.strip().lower() not in stpwords and word.strip() not in stopwords_list:
      try:
        if type(word)!=type(''):
          word=word.strip().lower().encode('utf-8')
        else:
          word=word.strip().lower()
        
        docembeddings.append(embeddingsmodel[lng][word])
        words.append(word)
        
      except :
        pass
    
  return [docembeddings,words]

directorypath='/home/ahmad/duplicate-detection/euronews/data/jsonfiles2/'
jsondata = [join(directorypath, f) for f in listdir(directorypath) if isfile(join(directorypath, f))]
shuffle(jsondata)

alllngpairs=[['en','ar'],['en','fa'],['en','fr'],['en','es'],['en','de'],['ar','fa'],['ar','fr'],['ar','es'],['ar','de'],['fa','fr'],['fa','es'],['fa','de'],['fr','es'],['fr','de'],['es','de']]
alllngpairs=[['en','ar']]

lngp = alllngpairs[0]
eventlist=[]
sentencelist=[]
enarticle=[]
foreignarticle=[]
enembeddings=[]
foreignembeddings=[]
enwords=[]
foreignwords=[]
for idx,filenm in enumerate(jsondata):
  with open(filenm,"r") as myfile:
    dayjson=json.load(myfile)
  
  for event in dayjson:
    if any([ll not in event.keys() for ll in lngp]):
        continue
    
    timenow=time.time()
    #text="PM Theresa May has struck a last-minute deal with the EU in a bid to move Brexit talks on to the next phase.    There will be no \"hard border\" with Ireland; and the rights of EU citizens in the UK and UK citizens in the EU will be protected.The so-called \"divorce bill\" will amount to between £35bn and £39bn, Downing Street sources say.The European Commission president said it was a \"breakthrough\" and he was confident EU leaders will approve it.".replace(" " ,"%20")
    onlineeventcoder_url="http://149.165.168.205:5123/process/sentence?sentence="+(event['en']['text']).replace("\n"," ").replace(" " ,"%20").encode('utf-8')
    rawevents = urllib2.urlopen(onlineeventcoder_url).read()
    jsonevents = json.loads(rawevents.decode("utf8"))
    if 'DUMMY_ID' in jsonevents.keys():
      eventsunordered=[jsonevents["DUMMY_ID"]["sents"][sent]["events"] for sent in jsonevents["DUMMY_ID"]["sents"].keys() if 'events' in jsonevents["DUMMY_ID"]["sents"][sent].keys()]
      sentencesunordered=[jsonevents["DUMMY_ID"]["sents"][sent]["content"] for sent in jsonevents["DUMMY_ID"]["sents"].keys() if 'events' in jsonevents["DUMMY_ID"]["sents"][sent].keys()]
    
    
    for evntidx in range(len(eventsunordered)):
      for subevnt in eventsunordered[evntidx]:
        eventlist.append(subevnt)
        sentencelist.append(sentencesunordered[evntidx])
        enarticle.append(event['en']['text'])
        foreignarticle.append(event['ar']['text'])
        mtrx,words=getembeddings(event['en']['text'],'en') #
        enembeddings.append(mtrx) #sentencesunordered[evntidx]
        enwords.append(words)
        mtrx,words=getembeddings(event['ar']['text'],'ar')
        foreignembeddings.append(mtrx)
        foreignwords.append(words)
  
  break

print dayjson[1]['ar']['text'].split(".")

len(foreignembeddings),len(foreignarticle)
del foreignembeddings[1]
del enembeddings[1]
foreignembeddings,enembeddings
eventlist[3]
sentencelist[3]
enarticle[3]
foreignarticle[1]

for ev,se,en,es in zip(eventlist,sentencelist,enarticle,foreignarticle):
  print ev
  print se
  print en
  print es
  print "--------------"

distance=[]
for enembdng,frnembdng in zip(enembeddings,foreignembeddings):
  distance.append(sklearn.metrics.pairwise.euclidean_distances(np.mean(enembdng,axis=0).reshape(1,-1),np.mean(frnembdng,axis=0).reshape(1,-1)))


puritydistance=[]
for enembdng,frnembdng,enwrds,frnwrds in zip(enembeddings,foreignembeddings,enwords,foreignwords):
  wordsdist,clustersdist,numclusters,pureclustersratio=dbclustering_purity(np.array(enembdng),np.array(frnembdng),enwrds,frnwrds,dbscan_eps=0.335, dbscan_minPts=2, min_samples_pt =2)
  #clustersdist,numclusters,pureclustersratio=dbclustering_purity(np.array(enembdng),np.array(frnembdng),dbscan_eps=0.335, dbscan_minPts=2, min_samples_pt =2)
  puritydistance.append(pureclustersratio)



negativeidx=[]
n=len(foreignembeddings)
for _ in range(5*n):
  negativeidx.append(list(np.random.choice(n,2, replace=False)))

negdistance=[]
for idx1,idx2 in negativeidx:
  negdistance.append(sklearn.metrics.pairwise.euclidean_distances(np.mean(enembeddings[idx1],axis=0).reshape(1,-1),np.mean(foreignembeddings[idx2],axis=0).reshape(1,-1)))



negpuritydistance=[]
for idx1,idx2 in negativeidx:
  wordsdist,clustersdist,numclusters,pureclustersratio=dbclustering_purity(np.array(enembeddings[idx1]),np.array(foreignembeddings[idx2]),enwords[idx1],foreignwords[idx2],dbscan_eps=0.335, dbscan_minPts=2, min_samples_pt =2)
  negpuritydistance.append(pureclustersratio)

print distance
print puritydistance
print negdistance
print negpuritydistance

idx1=1
idx2=0
print sklearn.metrics.pairwise.euclidean_distances(np.mean(enembeddings[idx1],axis=0).reshape(1,-1),np.mean(foreignembeddings[idx2],axis=0).reshape(1,-1))
wordsdist,clustersdist,numclusters,pureclustersratio=dbclustering_purity(np.array(enembeddings[idx1]),np.array(foreignembeddings[idx2]),enwords[idx1],foreignwords[idx2],dbscan_eps=0.335, dbscan_minPts=2, min_samples_pt =2)
print pureclustersratio

print sklearn.metrics.pairwise.euclidean_distances(np.mean(enembeddings[idx1],axis=0).reshape(1,-1),np.mean(foreignembeddings[idx2],axis=0).reshape(1,-1))
wordsdist,clustersdist,numclusters,pureclustersratio=dbclustering_purity(np.array(enembeddings[idx1]),np.array(foreignembeddings[idx2]),enwords[idx1],foreignwords[idx2],dbscan_eps=0.335, dbscan_minPts=2, min_samples_pt =2)
print pureclustersratio



correct=0
incorrect=0
for esidx,esembdng in enumerate(foreignembeddings):
  minidx=-1
  mindist=100
  for enidx,enembdng in enumerate(enembeddings):
    #clustersdist,numclusters,pureclustersratio=dbclustering_purity(np.array(enembdng),np.array(esembdng),dbscan_eps=0.335, dbscan_minPts=2, min_samples_pt =2)
    pureclustersratio=sklearn.metrics.pairwise.euclidean_distances(np.mean(enembdng,axis=0).reshape(1,-1),np.mean(esembdng,axis=0).reshape(1,-1))
    if mindist > pureclustersratio:
      minidx=enidx
      mindist=pureclustersratio
  if esidx==minidx:
    correct+=1
    print esidx
  else:
    incorrect+=1

print correct,incorrect


def dbclustering_purity(x1,x2,w1,w2,dbscan_eps=0.5, dbscan_minPts=2,min_samples_pt=2):
  
  if x1.size ==0 or x2.size ==0:
    return [[],[-100000,-100000,-100000], -100000]
  
  W=w1+w2
  X=np.vstack((x1,x2))
  X = StandardScaler().fit_transform(X)
  Y=[1]*x1.shape[0]+[2]*x2.shape[0]
  
  distance = cosine_similarity(X)+1
  distance = distance/np.max(distance)
  distance = 1 - distance
  db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, metric='precomputed', n_jobs=1).fit(distance.astype('float64'))
  #db = hdbscan.HDBSCAN(min_samples = min_samples_pt, min_cluster_size=dbscan_minPts, metric='precomputed').fit(distance.astype('float64'))
  #print distance
  def cos_metric(x, y):
    i, j = int(x[0]), int(y[0])# extract indices
    #print cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
    return cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
  
  #db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, n_jobs=1, metric=cos_metric).fit(X)
  #db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, n_jobs=-1).fit(X)
  labels_=list(db.labels_)
  #labels_=dbscan(X, eps=0.5, min_samples=5)[1]
  _n=len(set(labels_))
  if -1 in labels_:
    _n -= 1
  clusters= [[] for _ in range(_n)]
  wordsclusters= [[] for _ in range(_n)]
  n_pure_cl=0
  n_noise_cl=0
  n_mixed_cl=0
  n_pure_1=0
  n_pure_2=0
  for _idx,_lbl in enumerate(labels_):
    if _lbl==-1:
      n_noise_cl+=1
    else:
      clusters[_lbl].append(Y[_idx])
      wordsclusters[_lbl].append(W[_idx])
  
  for _lbl in clusters:
    if len(set(_lbl))>1:
      n_mixed_cl+=1
    else:
      n_pure_cl+=1
      if _lbl[0]==1:
        n_pure_1+=1
      elif _lbl[0]==2:
        n_pure_2+=1
  
  #print n_pure_1,n_pure_2,n_mixed_cl
  if min(n_pure_1+n_mixed_cl,n_pure_2+n_mixed_cl)==0:
    return [wordsclusters, clusters, [n_pure_1,n_pure_2,n_mixed_cl,n_noise_cl], 1.0] #[clusters, [n_pure_cl,n_mixed_cl,n_noise_cl], 1.0]
  else:
    return [wordsclusters, clusters, [n_pure_1,n_pure_2,n_mixed_cl,n_noise_cl], 1.0*min(n_pure_1,n_pure_2)/(min(n_pure_1,n_pure_2)+n_mixed_cl+0.00001)] 



