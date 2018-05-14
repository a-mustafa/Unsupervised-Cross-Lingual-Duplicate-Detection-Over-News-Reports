import time
import numpy as np
import sys
import random
import os
from os import listdir
from os.path import isfile, join
import re
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.cluster import dbscan
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
from scipy.spatial import distance
from scipy import spatial
import cPickle
import matplotlib.pyplot as plt
from mediameter.cliff import Cliff
from leven import levenshtein
from multiprocessing import Pool,cpu_count
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from fasttext import FastVector
from nltk import StanfordNERTagger
from google.cloud import translate
from googleapiclient.discovery import build


from fasttext import FastVector

def loadfasttextmodel(filename):
  filename='/home/ahmad/fastText_multilingual/'
  w2v=dict()
  #['en','es','zh','hr','de','fa','ar','fr']['es','en','de']
  for lng in ['en']:
    w2v[lng] = FastVector(vector_file=filename+'wiki.'+lng+'.vec')
    w2v[lng].apply_transform(filename+'alignment_matrices/'+lng+'.txt')
  
  return w2v



def dbclustering_purity(_w2vpairs,dbscan_eps=0.5, dbscan_minPts=2,min_samples_pt=2):
  
  if _w2vpairs[0].size ==0 or _w2vpairs[1].size ==0:
    return [[],[-100000,-100000,-100000], -100000]
  
  
  X=np.vstack((_w2vpairs[0],_w2vpairs[1]))
  X = StandardScaler().fit_transform(X)
  Y=[1]*_w2vpairs[0].shape[0]+[2]*_w2vpairs[1].shape[0]
  
  distance = cosine_similarity(X)+1
  distance = distance/np.max(distance)
  distance = 1 - distance
  db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, metric='precomputed', n_jobs=1).fit(distance.astype('float64'))
  
  def cos_metric(x, y):
    i, j = int(x[0]), int(y[0])# extract indices
    #print cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
    return cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
  
  labels_=list(db.labels_)
  #labels_=dbscan(X, eps=0.5, min_samples=5)[1]
  _n=len(set(labels_))
  if -1 in labels_:
    _n -= 1
  clusters= [[] for _ in range(_n)]
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
    return [clusters, [n_pure_cl,n_mixed_cl,n_noise_cl], 1.0]
  else:
    return [clusters, [n_pure_cl,n_mixed_cl,n_noise_cl], 1.0*min(n_pure_1,n_pure_2)/(min(n_pure_1,n_pure_2)+n_mixed_cl+0.00001)]


sum([True for _lng in lng if 'es' == _lng[0] and 'en' == _lng[1]])
sum([True for _score in score if _score>1])
print len(transallleftNE),len(transallrightNE),len(lbl),len(lng)
#embeddingsmodel=loadfasttextmodel('Path To Vectors')

unifiedw2vmodel=dict()
'''
Allsentpairs=[]
Alllangpairs=[]
Allisdup_labels=[]
posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
posfilenames = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
negfilenames = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]
to=min(len(posfilenames),len(negfilenames))
print to
frm=0
#cnt=0
for frm in range(to):
  labels,langpairs,sentpairs=create_w2v_pairs(unifiedw2vmodel,[posfilenames[frm]],[negfilenames[frm]])
  if len(labels)==0:
    continue
  
  #print "processing ",frm,len(w2vpairs),len(w2vpairs[0]), " pairs"
  if frm%50 == 0 and frm>0:
    print frm
    
  Allisdup_labels.extend(labels)
  Alllangpairs.extend(langpairs)
  Allsentpairs.extend(sentpairs)

len(Alllangpairs),len(Allsentpairs)

transallleft=[]
transallright=[]
label=[]
lng=[]

for _i in range(27296,len(Allsentpairs)):
  if len(transallleft)>10000: #out of 124,948
    print "NEXT", _i
    break
  
  if _i % 1000==0:
    print _i, len(transallleft)
  
  _sent=Allsentpairs[_i]
  try:
    if ((Alllangpairs[_i][0]=='es' or Alllangpairs[_i][0]=='de') and Alllangpairs[_i][1]=='en') or (Alllangpairs[_i][0]=='de' and Alllangpairs[_i][1]=='es'):
      translation=service.translations().list(source=Alllangpairs[_i][0],target='en',q=[_sent[0]],format='text').execute()
      #transwords=[transw['translatedText'].encode('utf-8') for transw in translation['translations']]
      transwordsl=translation['translations'][0]['translatedText'].encode('utf-8')
      
      
      if Alllangpairs[_i][1]!='en':
        translation=service.translations().list(source=Alllangpairs[_i][1],target='en',q=_sent[1],format='text').execute()
        #transwords=[transw['translatedText'].encode('utf-8') for transw in translation['translations']]
        transwordsr=translation['translations'][0]['translatedText'].encode('utf-8')
      else:
        transwordsr=_sent[1]
      
      transallleft.append(transwordsl)
      transallright.append(transwordsr)
      
      label.append(Allisdup_labels[_i])
      lng.append(Alllangpairs[_i])
  except:
    print sys.exc_info()[0]
    pass
'''



def dbclustering_purity(_w2vpairs,dbscan_eps=0.5, dbscan_minPts=2,min_samples_pt=2):
  
  if _w2vpairs[0].size ==0 or _w2vpairs[1].size ==0:
    return [[],[-100000,-100000,-100000], -100000]
  
  
  X=np.vstack((_w2vpairs[0],_w2vpairs[1]))
  X = StandardScaler().fit_transform(X)
  Y=[1]*_w2vpairs[0].shape[0]+[2]*_w2vpairs[1].shape[0]
  
  distance = cosine_similarity(X)+1
  distance = distance/np.max(distance)
  distance = 1 - distance
  db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, metric='precomputed', n_jobs=1).fit(distance.astype('float64'))
  
  def cos_metric(x, y):
    i, j = int(x[0]), int(y[0])# extract indices
    #print cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
    return cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
  
  labels_=list(db.labels_)
  #labels_=dbscan(X, eps=0.5, min_samples=5)[1]
  _n=len(set(labels_))
  if -1 in labels_:
    _n -= 1
  clusters= [[] for _ in range(_n)]
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
    return [clusters, [n_pure_cl,n_mixed_cl,n_noise_cl], 1.0]
  else:
    return [clusters, [n_pure_cl,n_mixed_cl,n_noise_cl], 1.0*min(n_pure_1,n_pure_2)/(min(n_pure_1,n_pure_2)+n_mixed_cl+0.00001)]



Allpureclustersratio=[]
dbscanlabels=[]
dbh=0.01
_lng0='de'
_lng1='es'
def run(dbh=0.40,_lng0='es',_lng1='en'):

Allpureclustersratio=[]
dbscanlabels=[]
for idx in range(len(label)):
  if lng[idx][0]!=_lng0 or lng[idx][1]!=_lng1:
    continue
  
  w2vmatrix1=[]
  wlist=[]
  wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', transallleft[idx])
  for word in wordslist:
    if '' !=word.strip() and word.strip().lower() not in stpwords:
      try:
        if type(word)!=type(''):
          word=word.strip().lower().encode('utf-8')
        else:
          word=word.strip().lower()
        
        w2vmatrix1.append(list(embeddingsmodel['en'][word]))
      except:
        #print sys.exc_info()[0]
        pass
  
  embeddingpr=[np.array(w2vmatrix1)]
  
  w2vmatrix2=[]
  wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', transallright[idx])
  for word in wordslist:
    if '' !=word.strip() and word.strip().lower() not in stpwords:
      try:
        if type(word)!=type(''):
          word=word.strip().lower().encode('utf-8')
        else:
          word=word.strip().lower()
        
        w2vmatrix2.append(list(embeddingsmodel['en'][word]))
      except:
        #print sys.exc_info()[0]
        pass
  
  embeddingpr.append(np.array(w2vmatrix2))
  
  if len(embeddingpr[0])==0 or len(embeddingpr[1])==0:
    print idx
    continue
  
  
  if idx%1000==0:
    print "processing ",idx
  
  clustersdist,numclusters,pureclustersratio=dbclustering_purity(embeddingpr,dbscan_eps=dbh, dbscan_minPts=2, min_samples_pt =2)
  Allpureclustersratio.append(pureclustersratio)
  dbscanlabels.append(label[idx])


  
  return [Allpureclustersratio, dbscanlabels]


dbh=0.2
_lng0='de'
_lng1='es'
Allpureclustersratio, dbscanlabels=run(dbh,_lng0,_lng1)

countpos=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==1 and _lng[0]==_lng0 and _lng[1]==_lng1])
countneg=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==0 and _lng[0]==_lng0 and _lng[1]==_lng1])

countpos=sum([True for _lbl in dbscanlabels if _lbl==1 ])
countneg=sum([True for _lbl in dbscanlabels if _lbl==0 ])

h=0.5
TP=sum([True for pp,_lbl in zip(Allpureclustersratio,dbscanlabels) if pp<=h and pp>=0 and _lbl==1])
FP=sum([True for pp,_lbl in zip(Allpureclustersratio,dbscanlabels) if pp<=h and pp>=0 and _lbl==0])
TN=sum([True for pp,_lbl in zip(Allpureclustersratio,dbscanlabels) if pp>h and pp>=0 and _lbl==0])
FN=sum([True for pp,_lbl in zip(Allpureclustersratio,dbscanlabels) if pp>h and pp>=0 and _lbl==1])

d=countpos-(TP+FN)
#FN+=d
d=countneg-(TN+FP)
#FP+=d


poserror=sum([True for pp,_lbl in zip(Allpureclustersratio,labels) if pp<0 and _lbl==1])
negerror=sum([True for pp,_lbl in zip(Allpureclustersratio,labels) if pp<0 and _lbl==0])
Precision=100.0*TP/(TP+FP+0.000001)
Recall=100.0*TP/(TP+FN+0.000001)
F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
print dbh,_lng0,_lng1,TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001)),h,poserror,negerror

print Precision,Recall
print 100.0*FP/(FP+TN),100.0*FN/(FN+TP)
print TP+TN+FP+FN,countpos,countneg

