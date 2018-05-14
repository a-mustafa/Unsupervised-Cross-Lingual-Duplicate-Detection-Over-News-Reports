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

#load fasttext model
embeddingsmodel=loadfasttextmodel('Path To Vectors')

#Arabic stopwords
from nltk.corpus import stopwords
stopwords_list = set(stopwords.words("/home/ahmad/duplicate-detection/arabicstopwords.txt"))

#Stopwords
stpwords=set(stopwords.words("spanish")+stopwords.words("english")+stopwords.words("german")+stopwords.words("french"))
stpwords=set(stopwords.words("spanish")+stopwords.words("english"))

#Function to convert text to Embeddings matrix
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

#Get dataset files, and shuffle them
directorypath='/home/ahmad/duplicate-detection/euronews/data/jsonfiles2/'
jsondata = [join(directorypath, f) for f in listdir(directorypath) if isfile(join(directorypath, f))]
shuffle(jsondata)

#To select doc pairs based on language
alllngpairs=[['en','ar'],['en','fa'],['en','fr'],['en','es'],['en','de'],['ar','fa'],['ar','fr'],['ar','es'],['ar','de'],['fa','fr'],['fa','es'],['fa','de'],['fr','es'],['fr','de'],['es','de']]
alllngpairs=[['en','ar']]

lngp = alllngpairs[0] #currently working on language pair lngp
eventlist=[] # stores events
sentencelist=[] # stores sentences associated with events
enarticle=[] # stores English articles
foreignarticle=[] # stores foreign language articles
enembeddings=[] # stores english Embeddings matrix
foreignembeddings=[] # stores foreign language Embeddings matrix
enwords=[] # stores English words that were converted to embeddings
foreignwords=[] # stores foreign language words that were converted to embeddings
for idx,filenm in enumerate(jsondata):
  with open(filenm,"r") as myfile:
    dayjson=json.load(myfile)
  
  for event in dayjson:
    if any([ll not in event.keys() for ll in lngp]):
        # Process language pair "lngp" only
        continue
    
    timenow=time.time()
    #text="PM Theresa May has struck a last-minute deal with the EU in a bid to move Brexit talks on to the next phase.    There will be no \"hard border\" with Ireland; and the rights of EU citizens in the UK and UK citizens in the EU will be protected.The so-called \"divorce bill\" will amount to between £35bn and £39bn, Downing Street sources say.The European Commission president said it was a \"breakthrough\" and he was confident EU leaders will approve it.".replace(" " ,"%20")
    
    #Get and Process PETR events
    onlineeventcoder_url="http://149.165.168.205:5123/process/sentence?sentence="+(event['en']['text']).replace("\n"," ").replace(" " ,"%20").encode('utf-8')
    rawevents = urllib2.urlopen(onlineeventcoder_url).read()
    jsonevents = json.loads(rawevents.decode("utf8"))
    if 'DUMMY_ID' in jsonevents.keys():
      if jsonevents["DUMMY_ID"]["sents"]!=None:
        eventsunordered=[jsonevents["DUMMY_ID"]["sents"][sent]["events"] for sent in jsonevents["DUMMY_ID"]["sents"].keys() if 'events' in jsonevents["DUMMY_ID"]["sents"][sent].keys()]
        sentencesunordered=[jsonevents["DUMMY_ID"]["sents"][sent]["content"] for sent in jsonevents["DUMMY_ID"]["sents"].keys() if 'events' in jsonevents["DUMMY_ID"]["sents"][sent].keys()]
    
    
    for evntidx in range(len(eventsunordered)):
      for subevnt in eventsunordered[evntidx]:
        eventlist.append(subevnt)
        sentencelist.append(sentencesunordered[evntidx])
        enarticle.append(event['en']['text'])
        foreignarticle.append(event['ar']['text'])
        mtrx,words=getembeddings(sentencesunordered[evntidx],'en') #
        enembeddings.append(mtrx) #event['en']['text']
        enwords.append(words)
        mtrx,words=getembeddings(event['ar']['text'],'ar')
        foreignembeddings.append(mtrx)
        foreignwords.append(words)
  
  
  if idx % 10 ==0:
    print idx

'''
print dayjson[1]['ar']['text'].split(".")

len(foreignembeddings),len(foreignarticle)
del foreignembeddings[1]
del enembeddings[1]
foreignembeddings,enembeddings
eventlist[3]
sentencelist[3]
enarticle[3]
foreignarticle[1]
'''
foreignarticle[4] in foreignarticle[5:]


for idd,fr in enumerate(foreignarticle):
  if fr in foreignarticle[idd+1:]:
    idn=idd+1+foreignarticle[idd+1:].index(fr)
    print idd,idn

for ev,se,en,es in zip(eventlist,sentencelist,enarticle,foreignarticle):
  print ev
  print se
  print en
  print es
  print "--------------"

#Baseline1: euclidean_distances between duplicate pairs
distance=[]
for enembdng,frnembdng in zip(enembeddings,foreignembeddings):
  distance.append(sklearn.metrics.pairwise.euclidean_distances(np.mean(enembdng,axis=0).reshape(1,-1),np.mean(frnembdng,axis=0).reshape(1,-1)))

#OurApproach: Clustering Distance between duplicate pairs
puritydistance=[]
mixedwordslist=[]
for enembdng,frnembdng,enwrds,frnwrds in zip(enembeddings,foreignembeddings,enwords,foreignwords):
  wordsdist,mixedwords,clustersdist,numclusters,pureclustersratio=dbclustering_purity(np.array(enembdng),np.array(frnembdng),enwrds,frnwrds,dbscan_eps=0.27, dbscan_minPts=2, min_samples_pt =2)
  #wordsdist,mixedwords,clustersdist,numclusters,pureclustersratio=dbclustering_purity(np.array(enembdng),np.array(frnembdng),enwrds,frnwrds,dbscan_eps=0.335, dbscan_minPts=2, min_samples_pt =2)
  #clustersdist,numclusters,pureclustersratio=dbclustering_purity(np.array(enembdng),np.array(frnembdng),dbscan_eps=0.335, dbscan_minPts=2, min_samples_pt =2)
  puritydistance.append(pureclustersratio)
  mixedwordslist.append(mixedwords)


#Sampling random pairs
negativeidx=[]
n=len(foreignembeddings)
for _ in range(2*n):
  negativeidx.append(list(np.random.choice(n,2, replace=False)))

print sentencelist[negativeidx[2][0]]
print foreignarticle[negativeidx[2][1]]
print sentencelist[negativeidx[1][0]]
print foreignarticle[negativeidx[1][1]]

idx1l=[]
idx2l=[]
for idx1,idx2 in negativeidx:
   if abs(idx1-idx2)==1:
     print idx1,idx2
   idx1l.append(min(idx1,idx2))
   idx2l.append(max(idx1,idx2))

for iddd,idx1ll in enumerate(idx1l):
  if idx1ll in idx1l[iddd+1:]:
    index1=idx1l[iddd+1:].index(idx1ll)+1+iddd
    if idx2l[index1] == idx2l[iddd]:
      print iddd,index1, idx1l[iddd], idx2l[iddd],idx1l[index1],idx2l[index1]

#Baseline1: euclidean_distances between non-duplicate pairs
negdistance=[]
for idx1,idx2 in negativeidx:
  negdistance.append(sklearn.metrics.pairwise.euclidean_distances(np.mean(enembeddings[idx1],axis=0).reshape(1,-1),np.mean(foreignembeddings[idx2],axis=0).reshape(1,-1)))


#OurApproach: Clustering Distance between non-duplicate pairs
negpuritydistance=[]
negmixedwordslist=[]
for idx1,idx2 in negativeidx:
  wordsdist,mixedwords,clustersdist,numclusters,pureclustersratio=dbclustering_purity(np.array(enembeddings[idx1]),np.array(foreignembeddings[idx2]),enwords[idx1],foreignwords[idx2],dbscan_eps=0.27, dbscan_minPts=2, min_samples_pt =2)
  negpuritydistance.append(pureclustersratio)
  negmixedwordslist.append(mixedwords)

print distance
print puritydistance
print negdistance
print negpuritydistance
nrgdist=negpuritydistance[0:len(puritydistance)]

idx1=1
idx2=0
print sklearn.metrics.pairwise.euclidean_distances(np.mean(enembeddings[idx1],axis=0).reshape(1,-1),np.mean(foreignembeddings[idx2],axis=0).reshape(1,-1))
wordsdist,clustersdist,numclusters,pureclustersratio=dbclustering_purity(np.array(enembeddings[idx1]),np.array(foreignembeddings[idx2]),enwords[idx1],foreignwords[idx2],dbscan_eps=0.335, dbscan_minPts=2, min_samples_pt =2)
print pureclustersratio

print sklearn.metrics.pairwise.euclidean_distances(np.mean(enembeddings[idx1],axis=0).reshape(1,-1),np.mean(foreignembeddings[idx2],axis=0).reshape(1,-1))
wordsdist,clustersdist,numclusters,pureclustersratio=dbclustering_purity(np.array(enembeddings[idx1]),np.array(foreignembeddings[idx2]),enwords[idx1],foreignwords[idx2],dbscan_eps=0.335, dbscan_minPts=2, min_samples_pt =2)
print pureclustersratio


from matplotlib import pyplot
bins = np.linspace(0, 2, 20)
pyplot.hist(negpuritydistance,bins, alpha=0.5, label='Negative distances', color = "blue")
pyplot.hist(puritydistance,bins, alpha=0.5, label='Positive distances', color = "red")
pyplot.legend(loc='upper right')
pyplot.savefig("/home/ahmad/duplicate-detection/hist2.pdf")
pyplot.gcf().clear()


negdistance1=[nn[0][0] for nn in negdistance]
distance1=[nn[0][0] for nn in distance]
pyplot.hist(negdistance1,bins, alpha=0.5, label='Negative distances', color = "blue")
pyplot.hist(distance1,bins, alpha=0.5, label='Positive distances', color = "red")
pyplot.legend(loc='upper right')
pyplot.savefig("/home/ahmad/duplicate-detection/basehist.pdf")
pyplot.gcf().clear()


TP=0
FN=0
for dist in puritydistance:
  if dist>=0.25:
    FN+=1
  else:
    TP+=1

TN=0
FP=0
for dist in nrgdist:
  if dist>=0.25:
    TN+=1
  else:
    FP+=1

TPR=100.0*TP/(TP+FN+0.000001)
TNR=100.0*TN/(TN+FP+0.000001)
FPR=100.0*FP/(FP+TN+0.000001)
FNR=100.0*FN/(TP+FN+0.000001)
print TP,TN,FP,FN
print TPR,TNR,FPR,FNR

# Get the closest sentence min(distance)
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


# Function to calculate distance between x1 and x2 matrices using clustering
def dbclustering_purity(x1,x2,w1,w2,dbscan_eps=0.5, dbscan_minPts=2,min_samples_pt=2):
  #x1 and x2 = embedding matices
  #w1,w2 = words 
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
  mixedwords= []
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
  
  for clstrid,_lbl in enumerate(clusters):
    if len(set(_lbl))>1:
      n_mixed_cl+=1
      mixedwords.append(wordsclusters[clstrid])
    else:
      n_pure_cl+=1
      if _lbl[0]==1:
        n_pure_1+=1
      elif _lbl[0]==2:
        n_pure_2+=1
  
  #print n_pure_1,n_pure_2,n_mixed_cl
  if min(n_pure_1+n_mixed_cl,n_pure_2+n_mixed_cl)==0:
    return [wordsclusters,mixedwords, clusters, [n_pure_1,n_pure_2,n_mixed_cl,n_noise_cl], 1.0] #[clusters, [n_pure_cl,n_mixed_cl,n_noise_cl], 1.0]
  else:
    return [wordsclusters,mixedwords, clusters, [n_pure_1,n_pure_2,n_mixed_cl,n_noise_cl], 1.0*min(n_pure_1,n_pure_2)/(min(n_pure_1,n_pure_2)+n_mixed_cl+0.00001)] 



