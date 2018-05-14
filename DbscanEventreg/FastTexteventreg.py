
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
import hdbscan

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
  w2v['fr'] = FastVector(vector_file=filename+'wiki.fr.vec')
  w2v['fr'].apply_transform(filename+'alignment_matrices/fr.txt')
  
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




def create_w2v_pairs(w2vmodel,allposfiles,allnegfiles,lgpr):
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
  #print("creating positive pairs:")
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
        if langcode[artcle['lang']] not in lgpr:
          continue
        
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        #wordslist=set(wordslist)
        w2vmatrix1=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower() and word.strip().lower() not in stpwords:
            try:
              if type(word)==type(''):
                word=word.strip().lower().decode('utf-8')
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
        if langcode[artcle['lang']] not in lgpr:
          continue
        
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        #wordslist=set(wordslist)
        w2vmatrix2=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower() and word.strip().lower() not in stpwords: #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              if type(word)==type(''):
                word=word.strip().lower().decode('utf-8')
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
          #if [lgroup1[x1],lgroup2[x2]] in [[lgpr[0],lgpr[1]],[lgpr[1],lgpr[0]]]:#[['es','de'],['de','es']]:
          #if [lgroup1[x1],lgroup2[x2]] in [['en','de'],['de','en'],['es','en'],['en','es']]:
          #if lgroup1[x1] not in ['en','es','de'] or lgroup2[x2] not in ['de','es','en']:
          #if lgroup1[x1] not in ['en','es'] or lgroup2[x2] not in ['es','en']:
          #  continue
          
          pospairs.append([np.array(group1[x1]),np.array(group2[x2])])
          poswordpairs.append([wgroup1[x1],wgroup2[x2]])
          poslangpairs.append([lgroup1[x1],lgroup2[x2]])
          posEntJaccardSim.append(jsonfile['meta']['entityJaccardSim'])
          possentpairs.append([sentgroup1[x1],sentgroup2[x2]])
      
      
      
      #sys.stdout.write("\r")
      #sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allposfiles)), 100*idx/len(allposfiles)))
      #sys.stdout.flush()
      
    #except Exception as e: print(e)
    except:
      pass
  
  
  #print('\ncreating negative pairs...')
  
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
        if langcode[artcle['lang']] not in lgpr:
          continue
        
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        #wordslist=set(wordslist)
        w2vmatrix1=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower() and word.strip().lower() not in stpwords: #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              if type(word)==type(''):
                word=word.strip().lower().decode('utf-8')
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
        if langcode[artcle['lang']] not in lgpr:
          continue
        
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        #wordslist=set(wordslist)
        w2vmatrix2=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower() and word.strip().lower() not in stpwords: #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and
            try:
              if type(word)==type(''):
                word=word.strip().lower().decode('utf-8')
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
          #if [lgroup1[x1],lgroup2[x2]] in [[lgpr[0],lgpr[1]],[lgpr[1],lgpr[0]]]:#[['es','de'],['de','es']]:
          #if [lgroup1[x1],lgroup2[x2]] in [['en','de'],['de','en'],['es','en'],['en','es']]:
          #if lgroup1[x1] not in ['en','es','de'] or lgroup2[x2] not in ['de','es','en']:
          #if lgroup1[x1] not in ['en','es'] or lgroup2[x2] not in ['es','en']:
          #  continue
          
          negpairs.append([np.array(group1[x1]),np.array(group2[x2])])
          negwordpairs.append([wgroup1[x1],wgroup2[x2]])
          neglangpairs.append([lgroup1[x1],lgroup2[x2]])
          negEntJaccardSim.append(jsonfile['meta']['entityJaccardSim'])
          negsentpairs.append([sentgroup1[x1],sentgroup2[x2]])
      
      
      #sys.stdout.write("\r")
      #sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allnegfiles)), 100*idx/len(allnegfiles)))
      #sys.stdout.flush()
    except:
      pass
  
  
  #print("\nShuffling...")
  #print len(pospairs),len(negpairs),len(poswordpairs),len(negwordpairs),len(poslangpairs),len(neglangpairs)
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

def dist_dbclustering_purity(a,dbscan_eps=0.5, dbscan_minPts=5,min_samples_pt=5):
  _w2vpairs=a[0]
  _wordspairs=a[1]
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


#def clustering_purity(_w2vpairs,_wordspairs,dbscan_eps=0.5, dbscan_minPts=5):
def dbclustering_purity(a,dbscan_eps=0.5, dbscan_minPts=5,min_samples_pt=5):
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
  db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, n_jobs=-1).fit(X)
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

def hdbclustering_purity(a,dbscan_eps=0.5, dbscan_minPts=2,min_samples_pt=2):
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
  
  distance = cosine_similarity(X)+1
  distance = distance/np.max(distance)
  #db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, metric='precomputed', n_jobs=1).fit(distance.astype('float64'))
  db = hdbscan.HDBSCAN(min_samples = min_samples_pt, min_cluster_size=dbscan_minPts, metric='precomputed').fit(distance.astype('float64'))
  #print distance
  def cos_metric(x, y):
    i, j = int(x[0]), int(y[0])# extract indices
    print cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
    return cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
  
  #db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, n_jobs=1, metric=cos_metric).fit(X)
  #db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, n_jobs=1).fit(X)
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


#unifiedw2vmodel=loadunifiedw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
#unifiedw2vmodel=loadunifiedw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/twelve.table4.multiSkip.size_512+w_5+it_10.normalized')

unifiedw2vmodel=w2vmodel
#unifiedw2vmodel=loadmultilingualw2vmodel('')
AllEntJaccardSim=[]
Allwordspairs=[]
Allsentpairs=[]
w2vpairsList=[]
Alllangpairs=[]
Allisdup_labels=[]
#Allclustersdist=[]
#Allnumclusters=[]
#Allpureclustersratio=[]
Allclustersdist=[]
Allnumclusters=[]
Allpureclustersratio=[]
posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
posfilenames = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
negfilenames = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]
to=min(len(posfilenames),len(negfilenames))
print to
frm=0
#cnt=0
for frm in range(to):
  w2vpairs,labels,wordspairs,langpairs,EntJaccardSim,sentpairs=create_w2v_pairs(unifiedw2vmodel,[posfilenames[frm]],[negfilenames[frm]],['en','es'])
  if len(w2vpairs)==0:
    continue
  
  #print "processing ",frm,len(w2vpairs),len(w2vpairs[0]), " pairs"
  if frm%50 == 0 and frm>0:
    print frm
    
  
  pool = Pool(processes=8)
  result_list = pool.map(dist_dbclustering_purity, zip(w2vpairs,wordspairs))
  pool.close()
  pool.join()
  Allclustersdist.extend([res[0] for res in result_list])
  Allnumclusters.extend([res[1] for res in result_list])
  Allpureclustersratio.extend([res[2] for res in result_list])
  
  #for _w2vpr,_wordspairs in zip(w2vpairs,wordspairs):
  #  clustersdist,numclusters,pureclustersratio=dbclustering_purity(zip(_w2vpr,_wordspairs),dbscan_eps=0.5, dbscan_minPts=4, min_samples_pt =2)
  #  clustersdist,numclusters,pureclustersratio=clustering_purity(zip(_w2vpr,_wordspairs),dbscan_eps=0.4, dbscan_minPts=4, min_samples_pt =2)
  #  Allclustersdist.append(clustersdist)
  #  Allnumclusters.append(numclusters)
  #  Allpureclustersratio.append(pureclustersratio)
  #  
  #  clustersdist,numclusters,pureclustersratio=hdbclustering_purity(zip(_w2vpr,_wordspairs),dbscan_eps=0.5, dbscan_minPts=2, min_samples_pt =2)
  #  Allclustersdist2.append(clustersdist)
  #  Allnumclusters2.append(numclusters)
  #  Allpureclustersratio2.append(pureclustersratio)
  
  Allisdup_labels.extend(labels)
  Alllangpairs.extend(langpairs)



print len(Allpureclustersratio),len(Allnumclusters),len(Allclustersdist),len(Allisdup_labels),len(Alllangpairs),len(Allpureclustersratio2),len(Allnumclusters2),len(Allclustersdist2)
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


print cnt
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


target=Allpureclustersratio
tp=tn=fp=fn=0
for lg in langposneg.keys():
  lg=lg[1:-1].strip().split(",")
  lg=[lg[0].strip()[1:-1],lg[1].strip()[1:-1]]
  P=[pp for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]]
  N=[pp for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]]
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
  TP=sum([True for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if pp<=h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  FP=sum([True for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if pp<=h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  TN=sum([True for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if pp>h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  FN=sum([True for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if pp>h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  poserror=sum([True for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if pp<0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  negerror=sum([True for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if pp<0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
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




target=Allpureclustersratio
tp=tn=fp=fn=0
for lg in langposneg.keys():
  lg=lg[1:-1].strip().split(",")
  lg=[lg[0].strip()[1:-1],lg[1].strip()[1:-1]]
  h=0.5
  TP=sum([True for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if pp<=h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  FP=sum([True for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if pp<=h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  TN=sum([True for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if pp>h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  FN=sum([True for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if pp>h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  poserror=sum([True for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if pp<0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  negerror=sum([True for pp,lbl,lng in zip(target,Allisdup_labels,Alllangpairs) if pp<0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
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


cPickle.dump(Allclustersdist, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllclustersdistFTxtAttro2.p', 'wb'))
cPickle.dump(Allnumclusters, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllnumclustersFTxtAttro2.p', 'wb'))
cPickle.dump(Allpureclustersratio, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllpureclustersratioFTxtAttro2.p', 'wb'))
cPickle.dump(Allclustersdist2, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllclustersdistFTxtHDBscanAttro2.p', 'wb'))
cPickle.dump(Allnumclusters2, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllnumclustersFTxtHDBscanAttro2.p', 'wb'))
cPickle.dump(Allpureclustersratio2, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AllpureclustersratioFTxtHDBscanAttro2.p', 'wb'))
cPickle.dump(Allisdup_labels, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Allisdup_labelsFTxtAttro2.p', 'wb'))
cPickle.dump(Alllangpairs, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AlllangpairsFTxtAttro2.p', 'wb'))
