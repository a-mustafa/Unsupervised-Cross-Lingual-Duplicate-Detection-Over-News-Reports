import time
import numpy as np
import sys
import random
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

def loadunifiedw2vmodel(filename):
  w2v=dict()
  #filename='fifty_nine.table5.multiCluster.m_1000+iter_10+window_3+min_count_5+size_40.normalized'
  #filename='/home/ahmad/duplicate-detection/multilingual-embedding/three.table4.multiSkip.iter_10+window_3+min_count_5+size_40.normalized'
  with open(filename, "r") as myfile:
    for line in myfile:
      lineparts=line.strip().split(":")
      wordvector=lineparts[1].split(" ")
      #w2v[lineparts[0]][wordvector[0]]=map(float,wordvector[1:])
      if type(wordvector[0])==type(''):
        w2v[wordvector[0].decode('utf-8')]=list(map(float,wordvector[1:]))
      else:
        w2v[wordvector[0]]=list(map(float,wordvector[1:]))
  
  return w2v

def loadmultilingualw2vmodel(filename):
  filename='/home/ahmad/fastText_multilingual/'
  w2v=dict()
  #w2v['fr'] = FastVector(vector_file=filename+'wiki.fr.vec')
  #w2v['fr'].apply_transform(filename+'alignment_matrices/fr.txt')
  
  w2v['en'] = FastVector(vector_file=filename+'wiki.en.vec')
  w2v['en'].apply_transform(filename+'alignment_matrices/en.txt')
  
  w2v['es'] = FastVector(vector_file=filename+'wiki.es.vec')
  w2v['es'].apply_transform(filename+'alignment_matrices/es.txt')
  
  w2v['zh'] = FastVector(vector_file=filename+'wiki.zh.vec')
  w2v['zh'].apply_transform(filename+'alignment_matrices/zh.txt')
  
  w2v['hr'] = FastVector(vector_file=filename+'wiki.hr.vec')
  w2v['hr'].apply_transform(filename+'alignment_matrices/hr.txt')
  
  w2v['de'] = FastVector(vector_file=filename+'wiki.de.vec')
  w2v['de'].apply_transform(filename+'alignment_matrices/de.txt')
  
  #en_vector = w2v['en']["cat"]
  #es_vector = w2v['es']["gato"]
  #print(FastVector.cosine_similarity(es_vector, en_vector))
  
  
  return w2v




def create_w2v_pairs(w2vmodel,allposfiles,allnegfiles):
  langcode={"eng":"en","spa":"es","deu":"de","zho":"zh","ita":"it","fra":"fr","rus":"ru","swe":"sv","nld":"nl","tur":"tr","jpn":"ja","por":"pt","ara":"ar","fin":"fi","ron":"ro","kor":"ko","hrv":"hr","tam":"","hun":"hu","slv":"sl","pol":"pl","srp":"sr","cat":"ca","ukr":"uk"}
  #w2vmodel= loadmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
  pairs=[]
  wpairs=[]
  langpairs=[]
  EntJaccardSim=[]
  sentpairs=[]
  labels = []
  pospairs=[]
  poswordpairs=[]
  possentpairs=[]
  poslangpairs=[]
  posEntJaccardSim=[]
  print("creating positive pairs:")
  for idx,Pfilenm in enumerate(allposfiles):
    try:
      with open(Pfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      lgroup1=[]
      group1=[]
      wgroup1=[]
      sentgroup1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        #wordslist=set(wordslist)
        w2vmatrix1=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower() and word.strip().lower() not in stpwords:
            try:
              if type(word)!=type(''):
                word=word.strip().lower().encode('utf-8')
              else:
                word=word.strip().lower()
              
              #w2vmatrix1.append(w2vmodel[word])
              w2vmatrix1.append(w2vmodel[langcode[artcle['lang']]][word])
              wlist.append(word.strip().lower())
            except:
              pass
        
        sentgroup1.append(artcle['body'])
        lgroup1.append(langcode[artcle['lang']])
        group1.append(np.array(w2vmatrix1))
        wgroup1.append(wlist)
      
      lgroup2=[]
      group2=[]
      wgroup2=[]
      sentgroup2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        #wordslist=set(wordslist)
        w2vmatrix2=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower() and word.strip().lower() not in stpwords: #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              if type(word)!=type(''):
                word=word.strip().lower().encode('utf-8')
              else:
                word=word.strip().lower()
              
              #w2vmatrix2.append(w2vmodel[word])
              w2vmatrix2.append(w2vmodel[langcode[artcle['lang']]][word])
              wlist.append(word.strip().lower())
            except:
              pass
        
        sentgroup2.append(artcle['body'])
        lgroup2.append(langcode[artcle['lang']])
        group2.append(np.array(w2vmatrix2))
        wgroup2.append(wlist)
      
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          if [lgroup1[x1],lgroup2[x2]] not in [['en','de'],['de','en'],['es','en'],['en','es'],['es','de'],['de','es']]:
            #if lgroup1[x1] not in ['en','es','de'] or lgroup2[x2] not in ['de','es','en']: ,['es','en'],['en','es']
            #if lgroup1[x1] not in ['en','es'] or lgroup2[x2] not in ['es','en']:
            continue
          
          pospairs.append([np.array(group1[x1]),np.array(group2[x2])])
          poswordpairs.append([wgroup1[x1],wgroup2[x2]])
          poslangpairs.append([lgroup1[x1],lgroup2[x2]])
          posEntJaccardSim.append(jsonfile['meta']['entityJaccardSim'])
          possentpairs.append([sentgroup1[x1],sentgroup2[x2]])
      
      
      
      sys.stdout.write("\r")
      sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allposfiles)), 100*idx/len(allposfiles)))
      sys.stdout.flush()
      
    except:
      pass
  
  
  print('\ncreating negative pairs...')
  
  neglangpairs=[]
  negpairs=[]
  negwordpairs=[]
  negEntJaccardSim=[]
  negsentpairs=[]
  for idx,Nfilenm in enumerate(allnegfiles):
    try:
      with open(Nfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      
      lgroup1=[]
      sentgroup1=[]
      group1=[]
      wgroup1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        #wordslist=set(wordslist)
        w2vmatrix1=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower() and word.strip().lower() not in stpwords: #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              if type(word)!=type(''):
                word=word.strip().lower().encode('utf-8')
              else:
                word=word.strip().lower()
              
              #w2vmatrix1.append(w2vmodel[word])
              w2vmatrix1.append(w2vmodel[langcode[artcle['lang']]][word])
              wlist.append(word)
            except:
              pass
        
        sentgroup1.append(artcle['body'])
        lgroup1.append(langcode[artcle['lang']])
        group1.append(np.array(w2vmatrix1))
        wgroup1.append(wlist)
      
      
      
      lgroup2=[]
      group2=[]
      wgroup2=[]
      sentgroup2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        #wordslist=set(wordslist)
        w2vmatrix2=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower() and word.strip().lower() not in stpwords: #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and
            try:
              if type(word)!=type(''):
                word=word.strip().lower().encode('utf-8')
              else:
                word=word.strip().lower()
              
              #w2vmatrix2.append(w2vmodel[word])
              w2vmatrix2.append(w2vmodel[langcode[artcle['lang']]][word])
              wlist.append(word.strip().lower())
            except:
              pass
        
        sentgroup2.append(artcle['body'])
        lgroup2.append(langcode[artcle['lang']])
        group2.append(np.array(w2vmatrix2))
        wgroup2.append(wlist)
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          if [lgroup1[x1],lgroup2[x2]] not in [['en','de'],['de','en'],['es','en'],['en','es'],['es','de'],['de','es']]:
            #if lgroup1[x1] not in ['en','es','de'] or lgroup2[x2] not in ['de','es','en']:
            #if lgroup1[x1] not in ['en','es'] or lgroup2[x2] not in ['es','en']:
            continue
          
          negpairs.append([np.array(group1[x1]),np.array(group2[x2])])
          negwordpairs.append([wgroup1[x1],wgroup2[x2]])
          neglangpairs.append([lgroup1[x1],lgroup2[x2]])
          negEntJaccardSim.append(jsonfile['meta']['entityJaccardSim'])
          negsentpairs.append([sentgroup1[x1],sentgroup2[x2]])
      
      
      sys.stdout.write("\r")
      sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allnegfiles)), 100*idx/len(allnegfiles)))
      sys.stdout.flush()
    except:
      pass
  
  
  print("\nShuffling...")
  print len(pospairs),len(negpairs),len(poswordpairs),len(negwordpairs),len(poslangpairs),len(neglangpairs)
  for pospair,poswpair,posspair,poslpair,posEntJaccSim in zip(pospairs,poswordpairs,possentpairs,poslangpairs,posEntJaccardSim):
    pairs.append(pospair)
    labels.append(1)
    wpairs.append(poswpair)
    langpairs.append(poslpair)
    EntJaccardSim.append(posEntJaccSim)
    sentpairs.append(posspair)
  
  for negpair,negwpair,negspair,neglpair,negEntJaccSim in zip(negpairs,negwordpairs,negsentpairs,neglangpairs,negEntJaccardSim):
    pairs.append(negpair)
    labels.append(0)
    wpairs.append(negwpair)
    langpairs.append(neglpair)
    EntJaccardSim.append(negEntJaccSim)
    sentpairs.append(negspair)
  
  
  return pairs,labels, wpairs, langpairs, EntJaccardSim,sentpairs


stpwords=set(stopwords.words("spanish")+stopwords.words("english")+stopwords.words("german")+stopwords.words("french"))


#def clustering_purity(_w2vpairs,_wordspairs,dbscan_eps=0.5, dbscan_minPts=5):
def clustering_purity(a,dbscan_eps=0.5, dbscan_minPts=5,min_samples_pt=5):
  _w2vpairs=[a[0][0],a[1][0]]
  _wordspairs=[a[0][1],a[1][1]]
  
  #_w2vpairs=w2vpr
  #_wordspairs=wordsprs
  
  if _w2vpairs[0].size ==0 or _w2vpairs[1].size ==0:
    return [[],[-100000,-100000,-100000], -100000]
  
  
  for _l1,l1 in enumerate(_wordspairs[0]):
    for _l2,l2 in enumerate(_wordspairs[1]):
      editdist=1.0*levenshtein(l1,l2)/min(len(l1),len(l2))
      if editdist<0.3 and editdist>0:
        _w2vpairs[0][_l1]=_w2vpairs[1][_l2]
  
  #return [[],[-100000,-100000,-100000], -100000]
  X=np.vstack((_w2vpairs[0],_w2vpairs[1]))
  X = StandardScaler().fit_transform(X)
  Y=[1]*_w2vpairs[0].shape[0]+[2]*_w2vpairs[1].shape[0]
  
  #distance = cosine_similarity(X)+1
  #distance = distance/np.max(distance)
  #db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, metric='precomputed', n_jobs=1).fit(distance.astype('float64'))
  #db = hdbscan.HDBSCAN(min_samples = min_samples_pt, min_cluster_size=dbscan_minPts, metric='precomputed').fit(distance.astype('float64'))
  #print distance
  def cos_metric(x, y):
    i, j = int(x[0]), int(y[0])# extract indices
    print cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
    return cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
  
  #db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, n_jobs=1, metric=cos_metric).fit(X)
  db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, n_jobs=1).fit(X)
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


#w2vmodel= loadmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
sum([True for _i1,_i2 in zip(Allisdup_labels[:ll/2],Allisdup_labels[ll/2:]) if _i1!=_i2])
sum([True for _i in range(ll/2) if Allisdup_labels[_i/2]!=Allisdup_labels[ll/2+_i]])

#w2vmodelpath='/home/ahmad/duplicate-detection/multilingual-embedding/twelve.table4.translation_invariance.size_512+window_3.normalized'
unifiedw2vmodel=loadunifiedw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
unifiedw2vmodel=loadunifiedw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/twelve.table4.translation_invariance.size_512+window_3.normalized')
#unifiedw2vmodel=loadunifiedw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/twelve.table4.multiSkip.size_512+w_5+it_10.normalized')
unifiedw2vmodel=loadmultilingualw2vmodel('')

en_vector = w2v['en']["spinach"]
es_vector = w2v['es']["gato"]
print(FastVector.cosine_similarity(es_vector, en_vector))
w2v['en'].apply_transform(filename+'alignment_matrices/en.txt')
w2v['es'].apply_transform(filename+'alignment_matrices/es.txt')

en_vector = w2v['en']["spinach"]
es_vector = w2v['es']["gato"]
print(FastVector.cosine_similarity(es_vector, en_vector))

allkeys=[dt.decode('utf-8') if type(dt)==type('') else dt for dt in unifiedw2vmodel.keys()]
index=[]
for _idx1 in range(16405,len(allkeys)):
  if _idx1 in index:
    continue
  
  for _idx2 in range(_idx1,len(allkeys)):
    l1 =allkeys[_idx1]
    l2 =allkeys[_idx2]
    editdist=1.0*levenshtein(l1,l2)/min(len(l1),len(l2))
    if editdist<0.3 and editdist>0:
      unifiedw2vmodel[l2]=unifiedw2vmodel[l1]
      index.append(_idx2) 


cPickle.dump(unifiedw2vmodel, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/unifiedw2vmodel.p', 'wb'))
#from itertools import izip
from joblib import Parallel, delayed
from multiprocessing import Pool


unifiedw2vmodel=loadunifiedw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
#unifiedw2vmodel=loadunifiedw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/twelve.table4.multiSkip.size_512+w_5+it_10.normalized')

unifiedw2vmodel=loadmultilingualw2vmodel('')

posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
allposfiles = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
allnegfiles = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]

AllEntJaccardSim=[]
Allwordspairs=[]
Allsentpairs=[]
w2vpairsList=[]

Alllangpairs=[]
Allisdup_labels=[]
#Allclustersdist=[]
#Allnumclusters=[]
Allpureclustersratio=[]
to=min(len(allposfiles),len(allnegfiles))
print to
frm=0
cnt=0
for frm in range(0,to-50,50):
  w2vpairs,labels,wordspairs,langpairs,EntJaccardSim,sentpairs=create_w2v_pairs(embeddingsmodel,allposfiles[frm:frm+50],allnegfiles[frm:frm+50])
  if len(w2vpairs)==0:
    continue
  
  print "processing ",frm,len(w2vpairs),len(w2vpairs[0]), " pairs"
  
  for _embeddingpr in w2vpairs:
      clustersdist,numclusters,pureclustersratio=dbclustering_purity(_embeddingpr,dbscan_eps=0.3, dbscan_minPts=2, min_samples_pt =2)
      Allpureclustersratio.append(pureclustersratio)
  
  Allisdup_labels.extend(labels)
  Alllangpairs.extend(langpairs)



len(Allpureclustersratio),len(Allisdup_labels),len(Alllangpairs)

tp=tn=fp=fn=0
for lg in [['en','es'],['es','en'],['de','en'],['en','de'],['es','de'],['de','es']]:
  #lg=lg[1:-1].strip().split(",")
  #lg=[lg[0].strip()[1:-1],lg[1].strip()[1:-1]]
  h=0.5
  TP=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<=h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  FP=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<=h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  TN=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp>h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  FN=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp>h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  poserror=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  negerror=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  tp+=TP
  tn+=TN
  fp+=FP
  fn+=FN
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
  F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
  print 0.3,lg[0],lg[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001)),h,poserror,negerror


precision=100.0*tp/(tp+fp+0.0001)
recall=100.0*tp/(tp+fn+0.0001)
f1=100.0*(2.0*tp)/((2.0*tp+1.0*fn+fp)+0.000001)
f2=100.0*(5.0*tp)/((5.0*tp+4.0*fn+fp)+0.000001)
print tp,tn,fp,fn,str(100.0*(tp+tn)/(tp+tn+fp+fn++0.000001)) + "," + str(f1)+", "+ str(f2)+", "+ str(precision)+", "+ str(recall)+ str(", ")+ str(100.0*tp/(tp+fn+0.0001))+ str(", ")+ str(100.0*tn/(tn+fp+0.0001))+ str(", ")+ str( 100.0*fp/(tn+fp+0.0001))+ str(", ")+ str( 100.0*fn/(tp+fn+0.0001))+", "+str((tp+fn))+", "+str((tn+fp))+", "+str((1.0*tp+fn)/(tn+fp+tp+fn+0.0001))


  w2vpairsList.extend(w2vpairs)
  Allwordspairs.extend(wordspairs)
  cPickle.dump(Allisdup_labels, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/o/Allisdup_labelsFTxtAttro2_'+str(cnt)+'.p', 'wb'))
  cPickle.dump(Alllangpairs, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/o/AlllangpairsFTxtAttro2_'+str(cnt)+'.p', 'wb'))
  cPickle.dump(w2vpairsList, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/o/w2vpairsListFTxtAttro2_'+str(cnt)+'.p', 'wb'))
  cPickle.dump(Allwordspairs, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/o/AllwordspairsFTxtAttro2_'+str(cnt)+'.p', 'wb'))
  cnt+=1
  Allwordspairs=[]
  w2vpairsList=[]
  AllEntJaccardSim=[]
  Allwordspairs=[]
  Allsentpairs=[]
  w2vpairsList=[]
  Alllangpairs=[]
  Allisdup_labels=[]
  Allclustersdist=[]
  Allnumclusters=[]
  Allpureclustersratio=[]


Allisdup_labels=[]
Allclustersdist=[]
Allnumclusters=[]
Allpureclustersratio=[]
Alllangpairs=[]
Allwordspairs=[]
w2vpairsList=[]
for _cnt in range(0,cnt):
  w2vpairsList=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/o/w2vpairsListFTxtAttro2_'+str(_cnt)+'.p', 'rb'))
  Allwordspairs=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/o/AllwordspairsFTxtAttro2_'+str(_cnt)+'.p', 'rb'))
  #Allisdup_labels.extend(cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Allisdup_labelsFTxtAttro2_'+str(_cnt)+'.p', 'rb')))
  #Alllangpairs.extend(cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AlllangpairsFTxtAttro2_'+str(_cnt)+'.p', 'rb')))
  #Allnumclusters=[]
  #Allpureclustersratio=[]
  #Allclustersdist=[]
  for w2vpr,_wordspairs in zip(w2vpairsList,Allwordspairs):
    clustersdist,numclusters,pureclustersratio=clustering_purity(zip(w2vpr,_wordspairs),dbscan_eps=0.5, dbscan_minPts=4, min_samples_pt =2)
    #clustersdist,numclusters,pureclustersratio=clustering_purity(zip(w2vpr,_wordspairs),dbscan_eps=0.4, dbscan_minPts=4, min_samples_pt =2)
    Allclustersdist.append(clustersdist)
    Allnumclusters.append(numclusters)
    Allpureclustersratio.append(pureclustersratio)
    
    if len(Allpureclustersratio) % 3000 == 0:
      print len(Allpureclustersratio), "out of" , len(w2vpairsList)


cPickle.dump(Allclustersdist, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/o/AllclustersdistFTxtAttro2.p', 'wb'))
cPickle.dump(Allnumclusters, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/o/AllnumclustersFTxtAttro2.p', 'wb'))
cPickle.dump(Allpureclustersratio, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/o/AllpureclustersratioFTxtAttro2.p', 'wb'))





  pool = Pool(processes=8)
  result_list = pool.map(clustering_purity, zip(w2vpairs,wordspairs))
  pool.close()
  pool.join()
  #result_list = map(clustering_purity, zip(w2vpairs,wordspairs))
  #result_list = Parallel(n_jobs=12)(delayed(clustering_purity)(i1,i2) for i1,i2 in zip(w2vpairs,wordspairs))
  clustersdist=[res[0] for res in result_list]
  numclusters=[res[1] for res in result_list]
  pureclustersratio=[res[2] for res in result_list]
  Allclustersdist.extend(clustersdist)
  Allnumclusters.extend(numclusters)
  Allpureclustersratio.extend(pureclustersratio)
  print len(Allpureclustersratio),len(Allnumclusters),len(Allclustersdist),len(Allisdup_labels),len(Alllangpairs)
  #if len(Allisdup_labels) > 10000:
  cPickle.dump(Allclustersdist, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-es/Allclustersdist40dAttro2_'+str(cnt)+'.p', 'wb'))
  cPickle.dump(Allnumclusters, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-es/Allnumclusters40dAttro2_'+str(cnt)+'.p', 'wb'))
  cPickle.dump(Allpureclustersratio, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-es/Allpureclustersratio40dAttro2_'+str(cnt)+'.p', 'wb'))
  cPickle.dump(Allisdup_labels, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-es/Allisdup_labels40dAttro2_'+str(cnt)+'.p', 'wb'))
  cPickle.dump(Alllangpairs, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-es/Alllangpairs40dAttro2_'+str(cnt)+'.p', 'wb'))
  cPickle.dump(w2vpairsList, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-es/w2vpairsList40dAttro2_'+str(cnt)+'.p', 'wb'))
  cPickle.dump(Allwordspairs, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-es/Allwordspairs40dAttro2_'+str(cnt)+'.p', 'wb'))
  cnt+=1
  print np.mean(Allpureclustersratio)
  Allwordspairs=[]
  w2vpairsList=[]
  AllEntJaccardSim=[]
  Allwordspairs=[]
  Allsentpairs=[]
  w2vpairsList=[]
  Alllangpairs=[]
  Allisdup_labels=[]
  Allclustersdist=[]
  Allnumclusters=[]
  Allpureclustersratio=[]


print "SAVED..."
NEdist=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/NEdistAttro2.p', 'rb'))
len(Allsentpairs)
#AllsentpairsAttro2.p

Allisdup_labels=[]
Allclustersdist=[]
Allnumclusters=[]
Allpureclustersratio=[]
Alllangpairs=[]
Allwordspairs=[]
w2vpairsList=[]
for cnt in range(0,10):
  w2vpairsList.extend(cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-de/w2vpairsListFTxtAttro2_'+str(cnt)+'.p', 'rb')))
  Allwordspairs.extend(cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-de/AllwordspairsFTxtAttro2_'+str(cnt)+'.p', 'rb')))
  Allisdup_labels.extend(cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-de/Allisdup_labelsFTxtAttro2_'+str(cnt)+'.p', 'rb')))
  Alllangpairs.extend(cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-de/AlllangpairsFTxtAttro2_'+str(cnt)+'.p', 'rb')))

  Allclustersdist.extend(cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-es/AllclustersdistFTxtAttro2_'+str(cnt)+'.p', 'rb')))
  Allnumclusters.extend(cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-es/AllnumclustersFTxtAttro2_'+str(cnt)+'.p', 'rb')))
  Allpureclustersratio.extend(cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/en-es/AllpureclustersratioFTxtAttro2_'+str(cnt)+'.p', 'rb')))



print len(Alllangpairs),len(Allisdup_labels),len(Allclustersdist),len(Allnumclusters),len(Allpureclustersratio),len(w2vpairsList),len(Allwordspairs)


num_cores = cpu_count()/2
Alllangpairs=[]
Allisdup_labels=[]
Allnumclusters=[]
Allpureclustersratio=[]
Allclustersdist=[]
posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
allposfiles = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
allnegfiles = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]
to=min(len(allposfiles),len(allnegfiles))
print to
frm=0

for frm in range(0,to-50,50):
  w2vpairs,labels,wordspairs,langpairs,EntJaccardSim,sentpairs=create_w2v_pairs(unifiedw2vmodel,allposfiles[frm:frm+10],allnegfiles[frm:frm+10])
  print "processing ",frm,len(w2vpairs),len(w2vpairs[0]), " pairs"
  if len(w2vpairs)==0:
    continue
  
Allisdup_labels.extend(labels)
Alllangpairs.extend(langpairs)
pool = Pool(processes=num_cores)
result_list = pool.map(clustering_purity, w2vpairs)
pool.close()
pool.join()
#result_list = Parallel(n_jobs=num_cores)(delayed(f)(i) for i in items)
clustersdist=[res[0] for res in result_list]
numclusters=[res[1] for res in result_list]
pureclustersratio=[res[2] for res in result_list]
Allclustersdist.extend(clustersdist)
Allnumclusters.extend(numclusters)
Allpureclustersratio.extend(pureclustersratio)


  for w2vpr,_wordspairs in zip(w2vpairs,wordspairs):
    s1=[dt.decode('utf-8') if type(dt)==type('') else dt for dt in _wordspairs[0]]
    s2=[dt.decode('utf-8') if type(dt)==type('') else dt for dt in _wordspairs[1]]
    for _l1,l1 in enumerate(_wordspairs[0]):
      for _l2,l2 in enumerate(_wordspairs[1]):
        editdist=1.0*levenshtein(l1,l2)/min(len(l1),len(l2))
        if editdist<0.3 and editdist>0:
          #print l1,l2,editdist
          w2vpr[0][_l1]=w2vpr[1][_l2]
    
    clustersdist,numclusters,pureclustersratio=clustering_purity(w2vpr)
    Allclustersdist.append(clustersdist)
    Allnumclusters.append(numclusters)
    Allpureclustersratio.append(pureclustersratio)

print len(Allpureclustersratio1),len(Allnumclusters1),len(Allclustersdist1)
#print len(AllEntJaccardSim),len(Allwordspairs),len(Alllangpairs),len(Allsentpairs),len(w2vpairsList),len(Allisdup_labels)
print len(Alllangpairs),len(Allisdup_labels),len(Allclustersdist),len(Allnumclusters),len(Allpureclustersratio)

print len(w2vpairs),len(wordspairs),len(labels)

cPickle.dump(Allclustersdist, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllclustersdistAttro2.p', 'wb'))
cPickle.dump(Allnumclusters, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllnumclustersAttro2.p', 'wb'))
cPickle.dump(Allpureclustersratio, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllpureclustersratioAttro2.p', 'wb'))

cPickle.dump(Allisdup_labels, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Allisdup_labelsAttro2.p', 'wb'))
cPickle.dump(Alllangpairs, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AlllangpairsAttro2.p', 'wb'))
print "SAVED..."


cPickle.dump(AllEntJaccardSim, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllEntJaccardSimAttro2.p', 'wb'))
cPickle.dump(Allwordspairs, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllwordspairsAttro2.p', 'wb'))
cPickle.dump(Allsentpairs, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllsentpairsAttro2.p', 'wb'))
cPickle.dump(w2vpairsList, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/w2vpairsListAttro2.p', 'wb'))

print "SAVED..."


Allnumclusters=[]
Allpureclustersratio=[]
Allclustersdist=[]
for w2vpr,_wordspairs in zip(w2vpairsList,Allwordspairs):
  s1=[dt.decode('utf-8') if type(dt)==type('') else dt for dt in _wordspairs[0]]
  s2=[dt.decode('utf-8') if type(dt)==type('') else dt for dt in _wordspairs[1]]
  for _l1,l1 in enumerate(s1):
    for _l2,l2 in enumerate(s2):
      editdist=1.0*levenshtein(l1,l2)/min(len(l1),len(l2))
      if editdist<0.3 and editdist>0:
        #print l1,l2,editdist
        w2vpr[0][_l1]=w2vpr[1][_l2]
  
  clustersdist,numclusters,pureclustersratio=clustering_purity(w2vpr,dbscan_eps=0.5, dbscan_minPts=5)
  clustersdist,numclusters,pureclustersratio=clustering_purity(zip(w2vpr,wordsprs),dbscan_eps=0.4, dbscan_minPts=4, min_samples_pt =2)
  Allclustersdist.append(clustersdist)
  Allnumclusters.append(numclusters)
  Allpureclustersratio.append(pureclustersratio)
  
  if len(Allpureclustersratio) % 3000 == 0:
    print len(Allpureclustersratio), "out of" , len(w2vpairsList)

import hdbscan
Allnumclusters=[]
Allpureclustersratio=[]
Allclustersdist=[]
for w2vpr,_wordspairs in zip(w2vpairsList,Allwordspairs):
  clustersdist,numclusters,pureclustersratio=clustering_purity(zip(w2vpr,_wordspairs),dbscan_eps=0.5, dbscan_minPts=4, min_samples_pt =2)
  #clustersdist,numclusters,pureclustersratio=clustering_purity(zip(w2vpr,_wordspairs),dbscan_eps=0.4, dbscan_minPts=4, min_samples_pt =2)
  Allclustersdist.append(clustersdist)
  Allnumclusters.append(numclusters)
  Allpureclustersratio.append(pureclustersratio)
  
  if len(Allpureclustersratio) % 3000 == 0:
    print len(Allpureclustersratio), "out of" , len(w2vpairsList)


cPickle.dump(Allclustersdist, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllclustersdistAttro2.p', 'wb'))
cPickle.dump(Allnumclusters, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllnumclustersAttro2.p', 'wb'))
cPickle.dump(Allpureclustersratio, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllpureclustersratioAttro2.p', 'wb'))


print len(Allclustersdist),len(Allnumclusters),len(Allpureclustersratio),len(Allwordspairs),len(Alllangpairs)
len(w2vpairsList),len(Allwordspairs)
#Allisdup_labels=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Allisdup_labelsAttro2.p', 'rb'))
#w2vpairsList=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/w2vpairsListAttro2.p', 'rb'))
#w2vpairsList=w2vpairs
#len(Allpureclustersratio),len(Allisdup_labels),len(w2vpairsList),len(Allwordspairs),len(Alllangpairs)
#for i,_ll,pure,lng in zip(range(len(Allisdup_labels)),Allisdup_labels,Allpureclustersratio,Alllangpairs):
#  if _ll == 1 and pure > 0.3 and lng[0]=='es' and lng[1]=='en':
#    break
i_s=[idx for idx,(pp,lbl,lng) in enumerate(zip(Allpureclustersratio2,Allisdup_labels,Alllangpairs)) if lbl==0 and lng[0]=='en' and lng[1]=='es']

i_s=[idx for idx,(pp,lbl,lng) in enumerate(zip(Allpureclustersratio,Allisdup_labels,Alllangpairs)) if lbl==1 and lng[0]=='en' and lng[1]=='es']

i=i_s[0]
print i
print Alllangpairs[i]
print Allisdup_labels[i]
print Allclustersdist[i]
print Allnumclusters[i]
print Allpureclustersratio[i]
print Allwordspairs[i][0]
print Allwordspairs[i][1]
print pratio[i]
w2vpr=w2vpairsList[i]
wordsprs=Allwordspairs[i]
s1=[dt.decode('utf-8') if type(dt)==type('') else dt for dt in Allwordspairs[i][0]]
s2=[dt.decode('utf-8') if type(dt)==type('') else dt for dt in Allwordspairs[i][1]]
for _l1,l1 in enumerate(s1):
  for _l2,l2 in enumerate(s2):
    editdist=1.0*levenshtein(l1,l2)/min(len(l1),len(l2))
    if editdist<0.3 and editdist>0:
      print l1,l2,editdist
      w2vpr[0][_l1]=w2vpr[1][_l2]

clustersdist,numclusters,pureclustersratio=clustering_purity(zip(w2vpr,wordsprs),dbscan_eps=0.4, dbscan_minPts=4, min_samples_pt =2)
print numclusters,pureclustersratio,clustersdist
X=np.vstack((w2vpr[0],w2vpr[1]))
Y=[1]*w2vpr[0].shape[0]+[2]*w2vpr[1].shape[0]
db = DBSCAN(eps=0.5, min_samples=3, n_jobs=2).fit(X)
labels_=list(db.labels_)
_n=len(set(labels_))
if -1 in labels_:
  _n -= 1

clusters= [[] for _ in range(_n)]
n_pure_cl=0
n_pure_1=0
n_pure_2=0
n_noise_cl=0
n_mixed_cl=0
for _idx,_lbl in enumerate(labels_):
  if _lbl==-1:
    n_noise_cl+=1
  else:
    clusters[_lbl].append(Y[_idx])

for _lbl in Allclustersdist[i]:
  if len(set(_lbl))>1:
    n_mixed_cl+=1
  else:
    n_pure_cl+=1
    if _lbl[0]==1:
      n_pure_1+=1
    elif _lbl[0]==2:
      n_pure_2+=1

print n_pure_cl,n_pure_1,n_pure_2,n_mixed_cl,n_noise_cl
1.0*min(n_pure_1,n_pure_2)/(min(n_pure_1,n_pure_2)+n_mixed_cl+.00001)
print Allwordspairs[i][0]
print Allwordspairs[i][1]

print Allsentpairs[i][0]
print Allsentpairs[i][1]

from leven import levenshtein

s1=[dt.decode('utf-8') if type(dt)==type('') else dt for dt in Allwordspairs[i][0]]
s2=[dt.decode('utf-8') if type(dt)==type('') else dt for dt in Allwordspairs[i][1]]
for l1 in s1:
  for l2 in s2:
    
    editdist=1.0*levenshtein(l1,l2)/min(len(l1),len(l2))
    if editdist<0.3 and editdist>0:
      print l1,l2,editdist

print unifiedw2vmodel['marihuana']
print unifiedw2vmodel['marijuana']

del Allisdup_labels
del AllEntJaccardSim
del Alllangpairs


Allnumclusters=[]
Allpureclustersratio=[]
Allclustersdist=[]
for lng,w2vpr,_wordspairs in zip(Alllangpairs,w2vpairsList,Allwordspairs):
  if lng[0]=='es' and lng[1]=='en':
    s1=[dt.decode('utf-8') if type(dt)==type('') else dt for dt in _wordspairs[0]]
    s2=[dt.decode('utf-8') if type(dt)==type('') else dt for dt in _wordspairs[1]]
    for _l1,l1 in enumerate(s1):
      for _l2,l2 in enumerate(s2):
        editdist=1.0*levenshtein(l1,l2)/min(len(l1),len(l2))
        if editdist<0.3 and editdist>0:
          #print l1,l2,editdist
          w2vpr[0][_l1]=w2vpr[1][_l2]
    
    clustersdist,numclusters,pureclustersratio=clustering_purity(w2vpr,dbscan_eps=0.5, dbscan_minPts=2)
    Allclustersdist.append(clustersdist)
    Allnumclusters.append(numclusters)
    Allpureclustersratio.append(pureclustersratio)
  else:
    Allclustersdist.append([[-1,-1]])
    Allnumclusters.append([-1,-1,-1])
    Allpureclustersratio.append(-1)
  
  if len(Allpureclustersratio) % 3000 == 0:
    print len(Allpureclustersratio), "out of" , len(w2vpairsList)


len(Allpureclustersratio),len(Allisdup_labels),len(w2vpairsList),len(Allwordspairs),len(Alllangpairs)

langposneg=dict()
for i,z in enumerate(zip(Allpureclustersratio3,Allisdup_labels)):
  p,l=z
  langpairstr=str(Alllangpairs[i])#str(langpr[i])
  #lng=langpairstr[2:-2].split(",")
  #if lng[0][:2]=='es' and lng[1][:-2]=='en':
  if langpairstr not in langposneg.keys():
    langposneg[langpairstr]=[[],[]]
  
  langposneg[langpairstr][l].append(p)

for _langposneg in langposneg.keys():
  pos=langposneg[_langposneg][1]
  neg=langposneg[_langposneg][0]
  print _langposneg,len(pos),len(neg)

_langposneg=langposneg.keys()[1]
print _langposneg

for _langposneg in langposneg.keys():
  pos=langposneg[_langposneg][1]
  neg=langposneg[_langposneg][0]
  plt.clf()
  plt.close()
  plt.cla()
  f, axarr = plt.subplots(2, sharex=True)
  axarr[0].hist(neg, weights=np.zeros_like(neg) + 1. / (len(neg)+.00001), color="blue")
  axarr[0].set_title("Neg"+ _langposneg +"(Purity)")
  axarr[1].hist(pos, weights=np.zeros_like(pos) + 1. / (len(pos)+.00001), color="red")
  axarr[1].set_title("Pos"+ _langposneg +"(Purity)")
  plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/figures/"+_langposneg.replace("'","").replace("[","").replace("]","").replace(",","").replace(" ","_")+"AllpureclustersratioHistEventReg11.pdf")





cPickle.dump(Allclustersdist, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllclustersdistAttro2.p', 'wb'))
cPickle.dump(Allnumclusters, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllnumclustersAttro2.p', 'wb'))
cPickle.dump(Allpureclustersratio, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllpureclustersratioAttro2.p', 'wb'))

del w2vpairsList
del Allclustersdist
del Allnumclusters
del Allpureclustersratio

#Allpureclustersratio=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllpureclustersratioAttro2.p', 'rb'))

from mediameter.cliff import Cliff
my_cliff = Cliff('http://10.176.148.123',8999)
Allsentpairs=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllsentpairsAttro2.p', 'rb'))
Allloclists1=[]
Allorgslist1=[]
Allpeoplelist1=[]
for i in range(len(Allsentpairs)):
  res=my_cliff.parseText(Allsentpairs[i][0])
  loclists=[]
  orgslist=[]
  peoplelist=[]
  if 'results' in res.keys():
      if 'organizations' in res['results'].keys():
          for org in res['results']['organizations']:
              if 'name' in org.keys():
                  orgslist.append(org['name'])
      
      if 'people' in res['results'].keys():
          for ppl in res['results']['people']:
              if 'name' in ppl.keys():
                  peoplelist.append(ppl['name'])
      
      if 'places' in res['results'].keys():
          if 'mentions' in res['results']['places']:
              for place in res['results']['places']['mentions']:
                  loclists.append(str(place['lon'])+"-"+str(place['lat']))
  
  Allloclists1.append(" ".join(set(loclists)))
  Allorgslist1.append(" ".join(set(orgslist)))
  Allpeoplelist1.append(" ".join(set(peoplelist)))

cPickle.dump(Allloclists1, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Allloclists1Attro2.p', 'wb'))
cPickle.dump(Allorgslist1, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Allorgslist1Attro2.p', 'wb'))
cPickle.dump(Allpeoplelist1, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Allpeoplelist1Attro2.p', 'wb'))


Allloclists2=[]
Allorgslist2=[]
Allpeoplelist2=[]
for i in range(len(Allsentpairs)):
  res=my_cliff.parseText(Allsentpairs[i][1])
  loclists=[]
  orgslist=[]
  peoplelist=[]
  if 'results' in res.keys():
      if 'organizations' in res['results'].keys():
          for org in res['results']['organizations']:
              if 'name' in org.keys():
                  orgslist.append(org['name'])
      
      if 'people' in res['results'].keys():
          for ppl in res['results']['people']:
              if 'name' in ppl.keys():
                  peoplelist.append(ppl['name'])
      
      if 'places' in res['results'].keys():
          if 'mentions' in res['results']['places']:
              for place in res['results']['places']['mentions']:
                  loclists.append(str(place['lon'])+"-"+str(place['lat']))
  
  Allloclists2.append(" ".join(set(loclists)))
  Allorgslist2.append(" ".join(set(orgslist)))
  Allpeoplelist2.append(" ".join(set(peoplelist)))


cPickle.dump(Allloclists2, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Allloclists2Attro2.p', 'wb'))
cPickle.dump(Allorgslist2, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Allorgslist2Attro2.p', 'wb'))
cPickle.dump(Allpeoplelist2, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Allpeoplelist2Attro2.p', 'wb'))


print len(Allloclists1),len(Allorgslist1),len(Allpeoplelist1)
print len(Allloclists2),len(Allorgslist2),len(Allpeoplelist2)

locs1=[]
for pidx in range(len(Allloclists1)):
  dt1=Allloclists1[pidx].split(" ")
  dt1 = filter(None, dt1)
  if len(dt1)>0:
    locs1.append(" ".join([dt.decode('utf-8') if type(dt)==type('') else dt for dt in dt1]))
  else:
    locs1.append("")

locs2=[]
for pidx in range(len(Allloclists2)):
  dt1=Allloclists2[pidx].split(" ")
  dt1 = filter(None, dt1)
  if len(dt1)>0:
    locs2.append(" ".join([dt.decode('utf-8') if type(dt)==type('') else dt for dt in dt1]))
  else:
    locs2.append("")

peoples1=[]
for pidx in range(len(Allpeoplelist1)):
  dt1=Allpeoplelist1[pidx].split(" ")
  dt1 = filter(None, dt1)
  if len(dt1)>0:
    peoples1.append(" ".join([dt.decode('utf-8') if type(dt)==type('') else dt for dt in dt1]))
  else:
    peoples1.append("")

peoples2=[]
for pidx in range(len(Allpeoplelist2)):
  dt1=Allpeoplelist2[pidx].split(" ")
  dt1 = filter(None, dt1)
  if len(dt1)>0:
    peoples2.append(" ".join([dt.decode('utf-8') if type(dt)==type('') else dt for dt in dt1]))
  else:
    peoples2.append("")

orgs1=[]
for pidx in range(len(Allorgslist1)):
  dt1=Allorgslist1[pidx].split(" ")
  dt1 = filter(None, dt1)
  if len(dt1)>0:
    orgs1.append(" ".join([dt.decode('utf-8') if type(dt)==type('') else dt for dt in dt1]))
  else:
    orgs1.append("")


orgs2=[]
for pidx in range(len(Allorgslist2)):
  dt1=Allorgslist2[pidx].split(" ")
  dt1 = filter(None, dt1)
  if len(dt1)>0:
    orgs2.append(" ".join([dt.decode('utf-8') if type(dt)==type('') else dt for dt in dt1]))
  else:
    orgs2.append("")


cPickle.dump(locs1, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/locs1Attro2.p', 'wb'))
cPickle.dump(peoples1, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/peoples1Attro2.p', 'wb'))
cPickle.dump(orgs1, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/orgs1Attro2.p', 'wb'))

cPickle.dump(locs2, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/locs2Attro2.p', 'wb'))
cPickle.dump(peoples2, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/peoples2Attro2.p', 'wb'))
cPickle.dump(orgs2, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/orgs2Attro2.p', 'wb'))

#
#

Allisdup_labels2=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Allisdup_labelsAttro2.p', 'rb'))

locs1=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/locs1Attro2.p', 'rb'))
peoples1=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/peoples1Attro2.p', 'rb'))
orgs1=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/orgs1Attro2.p', 'rb'))

locs2=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/locs2Attro2.p', 'rb'))
peoples2=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/peoples2Attro2.p', 'rb'))
orgs2=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/orgs2Attro2.p', 'rb'))

Location_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, binary=True, preprocessor = None, stop_words = None)#, max_features = 1000) #, ngram_range=(1, 1)
Location_features = Location_vectorizer.fit_transform(locs1+locs2)
Location_features = Location_features.toarray()

Person_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, binary=True, preprocessor = None, stop_words = None)#, max_features = 1000) #, ngram_range=(1, 1)
Person_features = Person_vectorizer.fit_transform(peoples1+peoples2)
Person_features = Person_features.toarray()

Orgs_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, preprocessor = None, binary=True, stop_words = None)#, max_features = 1000) #, ngram_range=(1, 1)
Orgs_features = Orgs_vectorizer.fit_transform(orgs1+orgs2)
Orgs_features = Orgs_features.toarray()

#cPickle.dump(Location_features, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Location_featuresAttro2.p', 'wb'))
#cPickle.dump(Person_features, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Person_featuresAttro2.p', 'wb'))
#cPickle.dump(Orgs_features, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Orgs_featuresAttro2.p', 'wb'))


#print Person_features.shape, Orgs_features.shape
n=len(locs1)
del locs1
del locs2
del orgs1
del orgs2
del peoples1
del peoples2
NEsfeatures1= np.hstack((Location_features[:n,],Person_features[:n,],Orgs_features[:n,]))
cPickle.dump(NEsfeatures1, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/NEsfeatures1Attro2.p', 'wb'))
del NEsfeatures1
NEsfeatures2= np.hstack((Location_features[n:,],Person_features[n:,],Orgs_features[n:,]))
cPickle.dump(NEsfeatures2, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/NEsfeatures2Attro2.p', 'wb'))

#NEsfeatures1= np.hstack((Person_features[:len(Allloclists1),],Orgs_features[:len(Allloclists1),]))
#NEsfeatures2= np.hstack((Person_features[len(Allloclists1):,],Orgs_features[len(Allloclists1):,]))

#NEsfeatures1= Person_features[:len(Allloclists1),]
#NEsfeatures2= Person_features[len(Allloclists1):,]

print Location_features.shape,Person_features.shape,Orgs_features.shape,NEsfeatures1.shape,NEsfeatures2.shape

from scipy import spatial
NEdist=[]
for i in range(n):
  l1= np.hstack((Person_features[i,],Orgs_features[i,]))#l1= np.hstack((Location_features[i,],Person_features[i,],Orgs_features[i,]))
  l2= np.hstack((Person_features[n+i,],Orgs_features[n+i,]))#l2= np.hstack((Location_features[n+i,],Person_features[n+i,],Orgs_features[n+i,]))
  if (sum(l1)>0) or (sum(l2)>0):
    NEdist.append(spatial.distance.jaccard(l1,l2))
  else:
    NEdist.append(-100000)




from scipy import spatial
NEdist=[]
for l1,l2 in zip(NEsfeatures1,NEsfeatures2):
  if (sum(l1)>0) or (sum(l2)>0):
    NEdist.append(spatial.distance.jaccard(l1,l2))
  else:
    NEdist.append(-100000)

cPickle.dump(NEdist, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/NEdistAttro2.p', 'wb'))
cPickle.dump(Location_features, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Location_featuresAttro2.p', 'wb'))
cPickle.dump(Person_features, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Person_featuresAttro2.p', 'wb'))
cPickle.dump(Orgs_features, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Orgs_featuresAttro2.p', 'wb'))


#
#
#
'''
sum(NEsfeatures1[i])
sum(NEsfeatures2[i])
i=514
i=1
print Allisdup_labels[i],Alllangpairs[i]
spatial.distance.jaccard(NEsfeatures1[i],NEsfeatures2[i])
print orgs[i],"\t",orgs[len(NEsfeatures1)+i]
print peoples[i],"\t",peoples[len(NEsfeatures1)+i]
print locs[i],"\t",locs[len(NEsfeatures1)+i]

for x1 in peoples[i].split(" "):
  for x2 in peoples[len(NEsfeatures1)+i].split(" "):
    if x1 == x2:
      print x1

for x1 in orgs[i].split(" "):
  for x2 in orgs[len(NEsfeatures1)+i].split(" "):
    if x1 == x2:
      print x1

for x1 in locs[i].split(" "):
  for x2 in locs[len(NEsfeatures1)+i].split(" "):
    if x1 == x2:
      print x1



for i,z in enumerate(zip(NEdist,Allisdup_labels)):
  if z[0]>0.99 and z[1]==1 and Alllangpairs[i]==['en','de']:# and i>514:
    print i
    break


len(np.where(NEdist==-100000)[0])
Locdist=[]
for l1,l2 in zip(Location_features[:len(Allloclists),],Location_features[len(Allloclists):,]):
  a=spatial.distance.jaccard(l1,l2)
  if np.isnan(a):
    Locdist.append(1)
  else:
    Locdist.append(a)


import cPickle
cPickle.dump(Allclustersdist, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllclustersdistAttro.p', 'wb'))
cPickle.dump(Allnumclusters, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllnumclustersAttro.p', 'wb'))
cPickle.dump(Allpureclustersratio, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllpureclustersratioAttro.p', 'wb'))
cPickle.dump(Allisdup_labels, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Allisdup_labelsAttro.p', 'wb'))
cPickle.dump(Alllangpairs, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AlllangpairsAttro.p', 'wb'))
'''
#obj = cPickle.load(open('save.p', 'rb'))
Allisdup_labels=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Allisdup_labelsAttro2.p', 'rb'))
len(Allisdup_labels)
Alllangpairs=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AlllangpairsAttro2.p', 'rb'))
Allpureclustersratio=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllpureclustersratioAttro2.p', 'rb'))
Allnumclusters=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllnumclustersAttro2.p', 'rb'))
Allclustersdist=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllclustersdistAttro2.p', 'rb'))

#
#
#


#
#### Histograms
#
#Allpureclustersratio
#Entropy
#Purity
#AllNEdist
#Allcosdist

import matplotlib.pyplot as plt


dataset=[]
duplabels=[]
pratio=[]
cosd=[]
langpr=[]
for clstring,lbl,_langpr in zip(Allclustersdist,Allisdup_labels,Alllangpairs):
  if len(clstring)==0:
    continue
  instance=[]
  for clstr in clstring.values():
    n_doc1=0
    n_doc2=0
    for doc in clstr:
      if doc==1:
        n_doc1+=1
      elif doc==2:
        n_doc2+=1
    
    instance.append(1.0*abs(n_doc1-n_doc2)/max(n_doc1,n_doc2))
  
  if len(instance)>0:
    pratio.append(1.0*sum([True for ones in instance if ones == 1.0])/len(instance))
    dataset.append(instance)
    langpr.append(_langpr)
    duplabels.append(lbl)

print len(dataset),len(duplabels),len(pratio),len(cosd)

pratio=[]
for clstring in Allnumclusters:
  pratio.append(1.0*clstring[1]/(clstring[0]+clstring[1]+0.00001))
  1.0*min(n_pure_1,n_pure_2)/(min(n_pure_1,n_pure_2)+n_mixed_cl+0.00001)


mratio=[]
for clstring in Allnumclusters:
  mratio.append(1.0*clstring[1]/(clstring[0]+clstring[1]+0.00001))


sum([True for mnmn in minmin if mnmn > 0])
overlap=[]
pratio=[]
minmin=[]
pure_1=[]
pure_2=[]
for clusters in Allclustersdist:
  n_mixed_cl=0
  n_pure_cl=0
  n_pure_1=0
  n_pure_2=0
  for clstr in clusters:
    if len(set(clstr))>1:
      n_mixed_cl+=1
    else:
      n_pure_cl+=1
      if clstr[0]==1:
        n_pure_1+=1
      elif clstr[0]==2:
        n_pure_2+=1
  
  minmin.append(min(n_pure_1,n_pure_2))
  pure_1.append(n_pure_1)
  pure_2.append(n_pure_2)
  overlap.append(1.0*n_mixed_cl/(min(n_pure_1+n_mixed_cl, n_pure_2+n_mixed_cl)+0.0001))
  if min(n_pure_1+n_mixed_cl,n_pure_2+n_mixed_cl)==0:
    pratio.append(1.0)
  else:
    pratio.append(1.0*min(n_pure_1,n_pure_2)/(min(n_pure_1,n_pure_2)+n_mixed_cl+0.00001))



lang=dict()
for ii in range(len(Alllangpairs)):
  
  strstr=str(Alllangpairs[ii])
  if strstr not in lang.keys():
    lang[strstr]=0
  
  lang[strstr]+=1

print lang
#
#
#AllEntJaccardSim
#Allpureclustersratio
#NEdist
#Purity
#Entropy
#duplabels
#pratio
#mratio
#langpr
print len(Allpureclustersratio),len(Allnumclusters),len(Allclustersdist),len(Allisdup_labels),len(Alllangpairs),len(NEdist)
cnt=0
langposneg=dict()
#for i,z in enumerate(zip(NEdist,Allisdup_labels)):
#for i,z in enumerate(zip(Allpureclustersratio,Allisdup_labels)):
for i,z in enumerate(zip(Allpureclustersratio,Allisdup_labels)):
  p,l=z
  langpairstr=str(Alllangpairs[i])#str(langpr[i])
  #langpairstr=str(langpr[i])
  if langpairstr not in langposneg.keys():
    langposneg[langpairstr]=[[],[]]
  
  #if np.isnan(p):
  #  continue
  
  if p<0:
    cnt+=1
  #  continue
  
  langposneg[langpairstr][l].append(p)



for _langposneg in langposneg.keys():
  pos=langposneg[_langposneg][1]
  neg=langposneg[_langposneg][0]
  print _langposneg,len(pos),len(neg)


_langposneg=langposneg.keys()[7]
for _langposneg in langposneg.keys():
  pos=langposneg[_langposneg][1]
  neg=langposneg[_langposneg][0]
  plt.clf()
  plt.close()
  plt.cla()
  f, axarr = plt.subplots(2, sharex=True)
  axarr[0].hist(neg, weights=np.zeros_like(neg) + 1. / (len(neg)+.00001), color="blue")
  axarr[0].set_title("Neg"+ _langposneg +"(Purity)")
  axarr[1].hist(pos, weights=np.zeros_like(pos) + 1. / (len(pos)+.00001), color="red")
  axarr[1].set_title("Pos"+ _langposneg +"(Purity)")
  plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/figures/"+_langposneg.replace("'","").replace("[","").replace("]","").replace(",","").replace(" ","_")+"AllpureclustersratioHistEventReg3.pdf")




#Allpureclustersratio

for i,z in enumerate(zip(NEdist,Allisdup_labels)):
  langpairstr=str(Alllangpairs[i])
  h=hdict[langpairstr]
  p,l=z
  if (p<h and l==1) or (p<h and l==0) or ( p>h and l==0) or (p>h and l==1):
    continue
  else:
    print i,p,h,l

len(Allisdup_labels)
#Allpureclustersratio
len(Allpureclustersratio),len(Allisdup_labels),len(Alllangpairs)
sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if not pp <0 and lbl==0 and lng[0]=='zh' and lng[1]=='en'])
print sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if lbl==1 and lng[0]=='zh' and lng[1]=='en'])
print sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if lbl==0 and lng[0]=='zh' and lng[1]=='en'])
h=0.5
TP=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<=h and lbl==1 and lng[0]=='es' and lng[1]=='en'])
FP=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<=h and lbl==0 and lng[0]=='es' and lng[1]=='en'])
TN=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp>h and lbl==0 and lng[0]=='es' and lng[1]=='en'])
FN=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp>h and lbl==1 and lng[0]=='es' and lng[1]=='en'])


[idx for idx,(pp,lbl,lng) in enumerate(zip(Allpureclustersratio,Allisdup_labels,Alllangpairs)) if pp<=0.1 and lbl==0 and lng[0]=='es' and lng[1]=='es']

#Allisdup_labels,Alllangpairs,Allpureclustersratio,overlap,pratio
lg=['en','en']
tp=tn=fp=fn=0
for lg in langposneg.keys():
  lg=lg[1:-1].strip().split(",")
  lg=[lg[0].strip()[1:-1],lg[1].strip()[1:-1]]
  P=[pp for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]]
  N=[pp for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]]
  pvalue=P+N
  lbls=[1]*len(P)+[0]*len(N)
  argssorted=np.argsort(pvalue)
  pvaluesorted=[pvalue[l] for l in argssorted ]
  lblssorted=[lbls[l] for l in argssorted ]
  maxF1=0
  besth=-1
  for _idx,h in enumerate(pvaluesorted):
    TP=sum(lbls[:_idx])
    FP=_idx-TP
    FN=sum(lbls[_idx:])
    F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
    if maxF1 < F1:
      maxF1=F1
      besth=h
  
  h=besth
  TP=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<=h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  FP=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<=h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  TN=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp>h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  FN=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp>h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  poserror=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  negerror=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  tp+=TP
  tn+=TN
  fp+=FP
  fn+=FN
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
  F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
  print lg[0],lg[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001)),h,poserror,negerror


precision=100.0*tp/(tp+fp+0.0001)
recall=100.0*tp/(tp+fn+0.0001)
f1=100.0*(2.0*tp)/((2.0*tp+1.0*fn+fp)+0.000001)
f2=100.0*(5.0*tp)/((5.0*tp+4.0*fn+fp)+0.000001)
print tp,tn,fp,fn,str(100.0*(tp+tn)/(tp+tn+fp+fn++0.000001)) + "," + str(f1)+", "+ str(f2)+", "+ str(precision)+", "+ str(recall)+ str(", ")+ str(100.0*tp/(tp+fn+0.0001))+ str(", ")+ str(100.0*tn/(tn+fp+0.0001))+ str(", ")+ str( 100.0*fp/(tn+fp+0.0001))+ str(", ")+ str( 100.0*fn/(tp+fn+0.0001))+", "+str((tp+fn))+", "+str((tn+fp))+", "+str((1.0*tp+fn)/(tn+fp+tp+fn+0.0001))


tp=tn=fp=fn=0
for lg in [['en','es'],['es','en'],['de','en'],['en','de'],['es','de'],['de','es']]:
  #lg=lg[1:-1].strip().split(",")
  #lg=[lg[0].strip()[1:-1],lg[1].strip()[1:-1]]
  h=0.5
  TP=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<=h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  FP=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<=h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  TN=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp>h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  FN=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp>h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  poserror=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  negerror=sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp<0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  tp+=TP
  tn+=TN
  fp+=FP
  fn+=FN
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
  F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
  print 0.4,lg[0],lg[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001)),h,poserror,negerror


precision=100.0*tp/(tp+fp+0.0001)
recall=100.0*tp/(tp+fn+0.0001)
f1=100.0*(2.0*tp)/((2.0*tp+1.0*fn+fp)+0.000001)
f2=100.0*(5.0*tp)/((5.0*tp+4.0*fn+fp)+0.000001)
print tp,tn,fp,fn,str(100.0*(tp+tn)/(tp+tn+fp+fn++0.000001)) + "," + str(f1)+", "+ str(f2)+", "+ str(precision)+", "+ str(recall)+ str(", ")+ str(100.0*tp/(tp+fn+0.0001))+ str(", ")+ str(100.0*tn/(tn+fp+0.0001))+ str(", ")+ str( 100.0*fp/(tn+fp+0.0001))+ str(", ")+ str( 100.0*fn/(tp+fn+0.0001))+", "+str((tp+fn))+", "+str((tn+fp))+", "+str((1.0*tp+fn)/(tn+fp+tp+fn+0.0001))


sum([True for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if lng[0]==lg[0] and lng[1]==lg[1]])

for lg in langposneg.keys():
  lg=lg[1:-1].strip().split(",")
  h=0.5

  print(lg[0],lg[1],maxF1,besth)



#
#
#
from scipy import stats
stats.describe(P)
stats.describe(N)




TN=sum([pp,lbl for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp>h and lbl==0 and lng[0]=='en' and lng[1]=='en'])
FN=sum([pp,lbl for pp,lbl,lng in zip(Allpureclustersratio,Allisdup_labels,Alllangpairs) if pp>h and lbl==1 and lng[0]=='en' and lng[1]=='en'])




hdict={"['zh', 'en']":.5, "['de', 'de']":.5, "['es', 'de']":.83, "['en', 'de']":.85, "['en', 'en']":.2, "['es', 'en']":.81, "['en', 'hr']":.5, "['es', 'es']":.15, "['en', 'es']":.85, "['de', 'es']":.8, "['de', 'en']":.8}
hdict={"['zh', 'en']":.5, "['de', 'de']":.5, "['es', 'de']":.17, "['en', 'de']":.1, "['en', 'en']":.2, "['es', 'en']":.2, "['en', 'hr']":.5, "['es', 'es']":.15, "['en', 'es']":.22, "['de', 'es']":.8, "['de', 'en']":.15} #For Entity jaccard Sim
hdict={"['zh', 'en']":.5, "['de', 'de']":.5, "['es', 'de']":.995, "['en', 'de']":.95, "['en', 'en']":.5, "['es', 'en']":.5, "['en', 'hr']":.5, "['es', 'es']":.5, "['en', 'es']":.96, "['de', 'es']":.5, "['de', 'en']":.5} #For Entity jaccard Sim
hdict={"['zh', 'en']":.5, "['de', 'de']":.5, "['es', 'de']":.5, "['en', 'de']":.5, "['en', 'en']":.5, "['es', 'en']":.5, "['en', 'hr']":.5, "['es', 'es']":.5, "['en', 'es']":.5, "['de', 'es']":.5, "['de', 'en']":.5} 
hdict={"['zh', 'en']":.5, "['de', 'de']":.5, "['es', 'de']":.995, "['en', 'de']":.98, "['en', 'en']":.5, "['es', 'en']":.5, "['en', 'hr']":.5, "['es', 'es']":.5, "['en', 'es']":.97, "['de', 'es']":.5, "['de', 'en']":.5} #For Entity jaccard Sim
hdict={"['zh', 'en']":1, "['de', 'de']":1, "['es', 'de']":1, "['en', 'de']":1, "['en', 'en']":1, "['es', 'en']":1, "['en', 'hr']":1, "['es', 'es']":1, "['en', 'es']":1, "['de', 'es']":1, "['de', 'en']":1} #For Entity jaccard Sim
Acc=dict()
Fa=TP=FP=FN=TN=0
FNs=[]
FPs=[]
for i,z in enumerate(zip(NEdist,Allisdup_labels)):
#for i,z in enumerate(zip(pratio,duplabels)):
  #if preddbscan1[i][0]>preddbscan1[i][1]:
  langpairstr=str(Alllangpairs[i])
  #if langpairstr in ["['zh', 'en']","['de', 'de']","['en', 'hr']","['en', 'en']","['es', 'es']"]:
  #if langpairstr not in ["['en', 'en']","['es', 'es']"]:
  #  continue
  #h=-1*hdict[langpairstr]
  h=hdict[langpairstr]
  if langpairstr not in Acc.keys():
    Acc[langpairstr]={'TP':0,'FP':0,'TN':0,'FN':0,'Fa':0}
  
  p,l=z
  #p=-1*p
  #if np.isnan(p):
  #  continue
  
  if p<=h and l==1:
    TP+=1
    Acc[langpairstr]['TP']+=1
  elif p<=h and l==0:
    FP+=1
    Acc[langpairstr]['FP']+=1
    FPs.append(i)
  elif p>h and l==0:
    TN+=1
    Acc[langpairstr]['TN']+=1
  elif p>h and l==1:
    FN+=1
    Acc[langpairstr]['FN']+=1
    FNs.append(i)
  else:
    Fa+=1
    Acc[langpairstr]['Fa']+=1

Precision=100.0*TP/(TP+FP+0.000001)
Recall=100.0*TP/(TP+FN+0.000001)
F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
print("TP",TP,"TN",TN,"FP",FP,"FN",FN)
print("\n Accuracy " + str(100.0*(TP+TN)/(TP+TN+FP+FN+Fa)) + "\n F1 " + str(F1)+"\n F2 "+ str(F2)+"\n Precision "+ str(Precision)+"\n Recall "+ str(Recall)+ str("\n TPR ")+ str(100.0*TP/(TP+FN))+ str("\n TNR ")+ str(100.0*TN/(TN+FP))+ str("\n FPR ")+ str( 100.0*FP/(TN+FP))+ str("\n FNR ")+ str( 100.0*FN/(TP+FN))+"\n n_pos "+str((TP+FN))+"\n n_neg "+str((TN+FP))+"\n positive ratio "+str((1.0*TP+FN)/(TN+FP)))
print(TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001)))

print("Lang, TP, TN, FP, FN, Accuracy , f1 , f2 , precision, recall, TPR, TNR, FPR, FNR, n_pos, n_neg, positive ratio")

for _acc in Acc.keys():
  tp=Acc[_acc]['TP']
  fp=Acc[_acc]['FP']
  tn=Acc[_acc]['TN']
  fn=Acc[_acc]['FN']
  fa=Acc[_acc]['Fa']
  precision=100.0*tp/(tp+fp+0.0001)
  recall=100.0*tp/(tp+fn+0.0001)
  f1=100.0*(2.0*tp)/((2.0*tp+1.0*fn+fp)+0.000001)
  f2=100.0*(5.0*tp)/((5.0*tp+4.0*fn+fp)+0.000001)
  #print "\n",_acc
  #print("tp",tp,"tn",tn,"fp",fp,"fn",fn)
  #print(" Accuracy " + str(100.0*(tp+tn)/(tp+tn+fp+fn)) + "\n f1 " + str(f1)+"\n f2 "+ str(f2)+"\n precision "+ str(precision)+"\n recall "+ str(recall)+ str("\n TPR ")+ str(100.0*tp/(tp+fn+0.0001))+ str("\n TNR ")+ str(100.0*tn/(tn+fp+0.0001))+ str("\n FPR ")+ str( 100.0*fp/(tn+fp+0.0001))+ str("\n FNR ")+ str( 100.0*fn/(tp+fn+0.0001))+"\n n_pos "+str((tp+fn))+"\n n_neg "+str((tn+fp))+"\n positive ratio "+str((1.0*tp+fn)/(tn+fp+0.0001)))
  
  print(_acc,fa,tp,tn,fp,fn,str(100.0*(tp+tn)/(tp+tn+fp+fn+fa)) + "," + str(f1)+", "+ str(f2)+", "+ str(precision)+", "+ str(recall)+ str(", ")+ str(100.0*tp/(tp+fn+0.0001))+ str(", ")+ str(100.0*tn/(tn+fp+0.0001))+ str(", ")+ str( 100.0*fp/(tn+fp+0.0001))+ str(", ")+ str( 100.0*fn/(tp+fn+0.0001))+", "+str((tp+fn))+", "+str((tn+fp))+", "+str((1.0*tp+fn)/(tn+fp+tp+fn+0.0001)))

'''
  Event regestry Accuracy:
 Accuracy 71.1284310403
 F1 0.706594731103
 F2 0.672063849068
 Precision 0.772770070931
 Recall 0.650859106529
 TPR 65.0859106529
 TNR 78.0566163545
 FPR 21.9433836455
 FNR 34.9140893471
 n_pos 18915
 n_neg 16497
 positive ratio 1.14657210402

'''
#
#
h=.50
TP=FP=FN=TN=0
FNs=[]
FPs=[]
#for i,z in enumerate(zip(Allcosdist,Allisdup_labels)):
for i,z in enumerate(zip(Locdist,Allisdup_labels)):
  #if preddbscan1[i][0]>preddbscan1[i][1]:
  #  continue
  p,l=z
  if p<h and l==1:
    TP+=1
  elif p<h and l==0:
    FP+=1
    FPs.append(i)
  elif p>h and l==0:
    TN+=1
  elif p>h and l==1:
    FN+=1
    FNs.append(i)

Precision=100.0*TP/(TP+FP)
Recall=100.0*TP/(TP+FN)
F1=100.0*(2.0*TP)/(2.0*TP+1.0*FN+FP)
F2=100.0*(5.0*TP)/(5.0*TP+4.0*FN+FP)
print("TP",TP,"TN",TN,"FP",FP,"FN",FN)
print("\n Accuracy " + str(100.0*(TP+TN)/(TP+TN+FP+FN)) + "\n F1 " + str(F1)+"\n F2 "+ str(F2)+"\n Precision "+ str(Precision)+"\n Recall "+ str(Recall)+ str("\n TPR ")+ str(100.0*TP/(TP+FN))+ str("\n TNR ")+ str(100.0*TN/(TN+FP))+ str("\n FPR ")+ str( 100.0*FP/(TN+FP))+ str("\n FNR ")+ str( 100.0*FN/(TP+FN))+"\n n_pos "+str((TP+FN))+"\n n_neg "+str((TN+FP))+"\n positive ratio "+str((1.0*TP+FN)/(TN+FP)))

#
#
pos=[]
neg=[]
#for i,z in enumerate(zip(Allpureclustersratio,Allisdup_labels)):
for i,z in enumerate(zip(Locdist,Allisdup_labels)):
  p,l=z
  #if p==-100000:
  #  continue
  if l==1:
    pos.append(p)
  elif l==0:
    neg.append(p)

plt.clf()
plt.close()
plt.cla()
plt.hist(neg, weights=np.zeros_like(neg) + 1. / len(neg), color="blue")
plt.title("Neg (purityRatio)")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/negdbscanRatiohistEventReg3.pdf")

plt.clf()
plt.close()
plt.cla()
plt.hist(pos, weights=np.zeros_like(pos) + 1. / len(pos), color="red")
plt.title("Pos (purityRatio)")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/posdbscanRatiohistEventReg.pdf")

plt.clf()
plt.close()
plt.cla()
f, axarr = plt.subplots(2, sharex=True)
axarr[0].hist(neg, weights=np.zeros_like(neg) + 1. / len(neg), color="blue")
axarr[0].set_title("Neg (locDis)")
axarr[1].hist(pos, weights=np.zeros_like(pos) + 1. / len(pos), color="red")
axarr[1].set_title("Pos (locDis)")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/locDisthistEventReg.pdf")

#
#
#
from fasttext import FastVector
fr_dictionary = FastVector(vector_file='wiki.fr.vec')
en_dictionary = FastVector(vector_file='wiki.en.vec')
es_dictionary = FastVector(vector_file='wiki.es.vec')
fr_vector = fr_dictionary["chat"]
en_vector = en_dictionary["cat"]
es_vector = es_dictionary["gato"]

en_vector = en_dictionary["shakespeare"]
es_vector = es_dictionary["shakespeare"]
en_vector = en_dictionary["committee"]
es_vector = es_dictionary["comité"]
print(FastVector.cosine_similarity(fr_vector, en_vector))
print(FastVector.cosine_similarity(es_vector, en_vector))

fr_dictionary.apply_transform('alignment_matrices/fr.txt')
en_dictionary.apply_transform('alignment_matrices/en.txt')
es_dictionary.apply_transform('alignment_matrices/es.txt')
print(FastVector.cosine_similarity(fr_vector, ru_vector))