# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:29:20 2016

@author: Ahmad
"""


from datetime import datetime,date,timedelta
from dateutil import parser

#from unbalanced_dataset import UnderSampler, NearMiss, CondensedNearestNeighbour, OneSidedSelection,NeighbourhoodCleaningRule, TomekLinks, ClusterCentroids, OverSampler, SMOTE,SMOTETomek, SMOTEENN, EasyEnsemble, BalanceCascade

import os
import json 
#from nltk.corpus import stopwords
import gensim
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
from sklearn.cluster import KMeans,SpectralClustering
from scipy.spatial import distance
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#import nltk
#import pandas as pd
from sklearn.decomposition.truncated_svd import TruncatedSVD 
import time
import smtplib
from sklearn.svm import SVC
#from sklearn.lda import LDA
from sklearn.metrics import fbeta_score, accuracy_score, precision_score,recall_score

import nltk
from nltk.stem.snowball import SnowballStemmer
import string
import sys
from sklearn.linear_model import SGDClassifier

import nltk.chunk
#from nltk.corpus import gazetteers

#fname='../atrocitiesdata/d2vmodel.model'
#model = gensim.models.Doc2Vec.load(fname)

sentences='/home/ahmad/duplicate-detection/atrocitiesdata/TextList3.txt'
dataText=[]
f = open(sentences, 'r')
for d in f:  
  dataText.append(d.decode('utf-8'))

sentences='/home/ahmad/duplicate-detection/atrocitiesdata/ClusterLabel3.txt'
ClusterLabel=[]
f = open(sentences, 'r')
for d in f:
  ClusterLabel.append(int(d.strip()))

sentences='/home/ahmad/duplicate-detection/atrocitiesdata/LocationsList4.txt'
LocationsList=[]
f = open(sentences, 'r')
for d in f:  
    LocationsList.append(d.decode('utf-8'))

sentences='/home/ahmad/duplicate-detection/atrocitiesdata/PersonsList3.txt'
PersonsList=[]
f = open(sentences, 'r')
for d in f:  
    PersonsList.append(d.decode('utf-8'))

sentences='/home/ahmad/duplicate-detection/atrocitiesdata/OrgsList3.txt'
OrgsList=[]
f = open(sentences, 'r')
for d in f:  
    OrgsList.append(d.decode('utf-8'))

sentences='/home/ahmad/duplicate-detection/atrocitiesdata/DatesList3.txt'
DatesList=[]
f = open(sentences, 'r')
for d in f:  
    DatesList.append(d.decode('utf-8'))

'''
sentences='/home/ahmad/duplicate-detection/atrocitiesdata/VictimList3.txt'
VictimList=[]
f = open(sentences, 'r')
for d in f:
  VictimList.append(d.decode('utf-8'))
'''

sentences='/home/ahmad/duplicate-detection/atrocitiesdata/KeysList4.txt'
KeysList=[]
f = open(sentences, 'r')
for d in f:  
  KeysList.append(d.decode('utf-8').strip())
  
'''
sentences='/home/ahmad/duplicate-detection/atrocitiesdata/FramenetList.txt'
FramenetList=[]
f = open(sentences, 'r')
for d in f:  
  FramenetList.append(d.decode('utf-8').strip())
'''

PersonsList.append(" ")
OrgsList.append(" ")
print len(dataText), len(DatesList), len(LocationsList), len(ClusterLabel), len(PersonsList), len(OrgsList), len(KeysList)

#print len(dataText), len(DatesList), len(LocationsList), len(ClusterLabel), len(PersonsList), len(OrgsList), len(VictimList), len(KeysList), len(FramenetList)

from nltk.stem.snowball import SnowballStemmer
import string
#path = '/opt/datacourse/data/parts'
#token_dict = {}
stemmer = SnowballStemmer("english")

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item).lower())
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def clustering_purity(w2vpairs,dbscan_eps=0.5, dbscan_minPts=5):
  numclusters=[]
  clustersdist=[]
  pureclustersratio=[]
  
  for pridx in range(len(w2vpairs)):
    if w2vpairs[pridx][0].size ==0 or w2vpairs[pridx][1].size ==0:
      numclusters.append([-100000,-100000,-100000])
      pureclustersratio.append(-100000)
      clustersdist.append([])
      continue
    
    X=np.vstack((w2vpairs[pridx][0],w2vpairs[pridx][1]))
    Y=[1]*w2vpairs[pridx][0].shape[0]+[2]*w2vpairs[pridx][1].shape[0]
    
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts).fit(X)
    ll=np.unique(db.labels_)
    n_pure_cl=0
    n_noise_cl=0
    n_mixed_cl=0
    pairclusters=[]
    for _ll in ll:
      if _ll == -1:
        n_noise_cl+=1
        continue
      
      idx=np.where(db.labels_==_ll)[0]
      lblsofcl=[Y[_idx] for _idx in idx]
      pairclusters.append(lblsofcl)
      if len(set(lblsofcl))>1:
        n_mixed_cl+=1
      else:
        n_pure_cl+=1
    
    clustersdist.append(pairclusters)
    numclusters.append([n_pure_cl,n_mixed_cl,n_noise_cl])
    pureclustersratio.append(1.0*n_pure_cl/(n_pure_cl+n_mixed_cl+0.00001))
  
  return numclusters, pureclustersratio, clustersdist


def distancetfidfpairs(tfidfpairs):
  cosdistance=[]
  for pridx in range(len(tfidfpairs)): 
    cosdistance.append(spatial.distance.cosine(tfidfpairs[pridx][0],tfidfpairs[pridx][1]))
    #cosine_similarity(pairs[pridx][0],pairs[pridx][1])
  
  return cosdistance


def jaccarddistancepairs(binarypairs):
  jaccarddistance=[]
  for pridx in range(len(binarypairs)):
    val=spatial.distance.jaccard(binarypairs[pridx][0],binarypairs[pridx][1])
    if np.isnan(val):
      jaccarddistance.append(0)
    else:
      jaccarddistance.append(val)
  
  return jaccarddistance

#
'''
dataText_vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = 'english', max_features = 10000, ngram_range=(1, 3))
dataText_features = dataText_vectorizer.fit_transform(FramenetList)
dataText_features = dataText_features.toarray()

dataText_featuresList=[]
for i in range(len(model.docvecs)):
    dataText_featuresList.append(model.docvecs[i])

dataText_features=np.asarray(dataText_featuresList)
'''
dataText_vectorizer = TfidfVectorizer(analyzer = "word",tokenizer = tokenize, preprocessor = None, stop_words = 'english', max_features = 1000) #, ngram_range=(1, 1)
dataText_features = dataText_vectorizer.fit_transform(dataText)
dataText_features = dataText_features.toarray()

#d = distance.pdist(dataText_features[Trmask,:], metric='cosine')
#dataText_dist_Mat = distance.squareform(d)
#dataText_Sim_Mat=dataText_dist_Mat
#dataText_Sim_Mat[np.isnan(dataText_Sim_Mat)]=0

Location_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, binary=True, preprocessor = None, stop_words = None, max_features = 1000) #, ngram_range=(1, 1)
Location_features = Location_vectorizer.fit_transform(LocationsList)
Location_features = Location_features.toarray()
#d = distance.pdist(Location_features[Trmask,:], metric='jaccard')
#Location_dist_Mat = distance.squareform(d)
#Location_Sim_Mat=1-Location_dist_Mat
#Location_Sim_Mat[np.isnan(Location_Sim_Mat)]=0

Person_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, binary=True, preprocessor = None, stop_words = None, max_features = 1000) #, ngram_range=(1, 1)
Person_features = Person_vectorizer.fit_transform(PersonsList)
Person_features = Person_features.toarray()
#d = distance.pdist(Person_features[Trmask,:], metric='jaccard')
#Person_dist_Mat = distance.squareform(d)
#Person_Sim_Mat=1-Person_dist_Mat
#Person_Sim_Mat[np.isnan(Person_Sim_Mat)]=0

Orgs_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, preprocessor = None, binary=True, stop_words = None, max_features = 1000) #, ngram_range=(1, 1)
Orgs_features = Orgs_vectorizer.fit_transform(OrgsList)
Orgs_features = Orgs_features.toarray()
#d = distance.pdist(Orgs_features[Trmask,:], metric='jaccard')
#Orgs_dist_Mat = distance.squareform(d)
#Orgs_Sim_Mat=1-Orgs_dist_Mat
#Orgs_Sim_Mat[np.isnan(Orgs_Sim_Mat)]=0

Dates_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, preprocessor = None, binary=True, stop_words = None, max_features = 1000) #, ngram_range=(1, 1)
Dates_features = Dates_vectorizer.fit_transform(DatesList)
Dates_features = Dates_features.toarray()
#d = distance.pdist(Dates_features[Trmask,:], metric='jaccard')
#Dates_dist_Mat = distance.squareform(d)
#Dates_Sim_Mat=1-Dates_dist_Mat
#Dates_Sim_Mat[np.isnan(Dates_Sim_Mat)]=0


w2vmodelpath='/home/ahmad/duplicate-detection/multilingual-embedding/twelve.table4.translation_invariance.window_3+size_40.normalized'
w2vmodel=loadmodel(w2vmodelpath)
en_stop = set(stopwords.words('english'))
w2vdata=[]
wordsdata=[]
for artcle in dataText:
  wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle)
  #wordslist=set(wordslist)
  w2vmatrix=[]
  wlist=[]
  for word in wordslist:
    if '' !=word.strip().lower() and word.strip().lower() not in en_stop:
      try:
        #w2vmatrix.append(w2vmodel[word.strip().lower()])
        w2vmatrix.append(w2vmodel['en'][word.strip().lower()])
        wlist.append(word.strip().lower())
      except:
        pass
  
  w2vdata.append(np.array(w2vmatrix))
  wordsdata.append(wlist)


from dateutil import parser
#rootdir='D:\\NLP\\Mass dataset\\OrderedDocuments\\'
rootdir="../atrocitiesdata/OrderedDocuments"
dirsList=next(os.walk(rootdir))[1]
for d in range(len(dirsList)):
    dirsList[d]= datetime.strptime(dirsList[d], '%d_%b_%Y')

dirsList.sort()
dirsList=[os.path.join(rootdir, date.strftime(s,'%d_%b_%Y')) for s in dirsList]

n=dataText_features.shape[0]
#clf = SGDClassifier(random_state=100)
clf = SGDClassifier(loss='log', random_state=123456)
#clf = SGDClassifier()
nTrainingdays=10


Allnumclusters=[]
Allpureclustersratio=[]
Allclustersdist=[]
Allcosdist=[]
Allisdup_labels=[]
Trainingindexes=[]
AllLocationdist=[]
AllOrgsdist=[]
AllPersondist=[]
AllDatedist=[]

for day in range(len(dirsList)):
    if day > nTrainingdays:
        break
    fs=next(os.walk(dirsList[day]))[2]
    for i, j in enumerate(fs):
        key=j.split('.')[0]
        if key in KeysList:
          instanceindex=KeysList.index(key)
          Trainingindexes.append(instanceindex)
    
    if day==nTrainingdays:
      Trmask=np.zeros(n,dtype='bool')
      Trmask[Trainingindexes]=True
      datax=dataText_features[Trmask,:]
      Locations=Location_features[Trmask,:]
      Persons=Person_features[Trmask,:]
      Orgs=Orgs_features[Trmask,:]
      Dates=Dates_features[Trmask,:]
      
      documents = [w2vdata[i] for i in Trainingindexes]
      docterms = [wordsdata[i] for i in Trainingindexes]
      clusterslabels=[ClusterLabel[i] for i in Trainingindexes]
      
      Tfidfpairs=[]
      w2vpairs=[]
      Locationspairs=[]
      Personspairs=[]
      Orgspairs=[]
      Datespairs=[]
      isdup_labels=[]
      wordpairs=[]
      
      for x1 in range(len(documents)):
        for x2 in range(x1,len(documents)):
          w2vpairs.append([documents[x1],documents[x2]])
          Tfidfpairs.append([datax[x1],datax[x2]])
          wordpairs.append([docterms[x1],docterms[x2]])
          Locationspairs.append([Locations[x1],Locations[x2]])
          Personspairs.append([Persons[x1],Persons[x2]])
          Orgspairs.append([Orgs[x1],Orgs[x2]])
          Datespairs.append([Dates[x1],Dates[x2]])
          if clusterslabels[x1]==clusterslabels[x2]:
            isdup_labels.append(1)
          else:
            isdup_labels.append(0)
          
          if cnt==254:
            print x1,x2
          
          cnt+=1
      
      
      numclusters,pureclustersratio,clustersdist=clustering_purity(w2vpairs,dbscan_eps=0.5, dbscan_minPts=5)
      Allcosdist.extend(distancetfidfpairs(Tfidfpairs))
      AllLocationdist.extend(jaccarddistancepairs(Locationspairs))
      AllPersondist.extend(jaccarddistancepairs(Personspairs))
      AllOrgsdist.extend(jaccarddistancepairs(Orgspairs))
      AllDatedist.extend(jaccarddistancepairs(Datespairs))
      Allclustersdist.extend(clustersdist)
      Allnumclusters.extend(numclusters)
      Allpureclustersratio.extend(pureclustersratio)
      Allisdup_labels.extend(isdup_labels)
      
      Trainingindexes=[]
      

print len(Allcosdist),len(AllLocationdist),len(AllPersondist),len(AllOrgsdist),len(AllDatedist),len(Allpureclustersratio),len(Allisdup_labels)
trinstances=np.vstack((Allcosdist,AllLocationdist,AllPersondist,AllOrgsdist,AllDatedist,Allpureclustersratio)).T
#clf.fit(trinstances, trLabelsList)
np.where(trinstances==np.nan)
clf.partial_fit(trinstances, Allisdup_labels,np.unique(Allisdup_labels))
np.random.seed(123456) #12 #1234
countTrainingInstances=0
confh=0.95
minh=0.03
allpred=[]
alltrue=[]
allconf=[]
dayindexes=[]

for day in range(nTrainingdays,len(dirsList)):
    fs=next(os.walk(dirsList[day]))[2]
    for i, j in enumerate(fs):
        key=j.split('.')[0]
        if key in KeysList:
            instanceindex=KeysList.index(key)
            dayindexes.append(instanceindex)
    
    if day % 5 == 0: #if len(dayindexes)>1:
        Trmask=np.zeros(n,dtype='bool')
        Trmask[dayindexes]=True
        
        datax=dataText_features[Trmask,:]
        Locations=Location_features[Trmask,:]
        Persons=Person_features[Trmask,:]
        Orgs=Orgs_features[Trmask,:]
        Dates=Dates_features[Trmask,:]
        
        documents = [w2vdata[i] for i in dayindexes]
        docterms = [wordsdata[i] for i in dayindexes]
        clusterslabels=[ClusterLabel[i] for i in dayindexes]
        Tfidfpairs=[]
        w2vpairs=[]
        Locationspairs=[]
        Personspairs=[]
        Orgspairs=[]
        Datespairs=[]
        isdup_labels=[]
        wordpairs=[]
        for x1 in range(len(documents)):
          for x2 in range(x1,len(documents)):
            w2vpairs.append([documents[x1],documents[x2]])
            Tfidfpairs.append([datax[x1],datax[x2]])
            wordpairs.append([docterms[x1],docterms[x2]])
            Locationspairs.append([Locations[x1],Locations[x2]])
            Personspairs.append([Persons[x1],Persons[x2]])
            Orgspairs.append([Orgs[x1],Orgs[x2]])
            Datespairs.append([Dates[x1],Dates[x2]])
            if clusterslabels[x1]==clusterslabels[x2]:
              isdup_labels.append(1)
            else:
              isdup_labels.append(0)
        
        numclusters,pureclustersratio,clustersdist=clustering_purity(w2vpairs,dbscan_eps=0.5, dbscan_minPts=5)
        tfidfdist=distancetfidfpairs(Tfidfpairs)
        locdist=jaccarddistancepairs(Locationspairs)
        persondist=jaccarddistancepairs(Personspairs)
        orgdist=jaccarddistancepairs(Orgspairs)
        datesdist=jaccarddistancepairs(Datespairs)
        Allcosdist.extend(tfidfdist)
        AllLocationdist.extend(locdist)
        AllPersondist.extend(persondist)
        AllOrgsdist.extend(orgdist)
        AllDatedist.extend(datesdist)
        Allclustersdist.extend(clustersdist)
        Allnumclusters.extend(numclusters)
        Allpureclustersratio.extend(pureclustersratio)
        Allisdup_labels.extend(isdup_labels)
        dayinstances=np.vstack((tfidfdist,locdist,persondist,orgdist,datesdist,pureclustersratio)).T
        #dayinstances=np.vstack((Location_Sim_Mat[tr1,tr2], Person_Sim_Mat[tr1,tr2], Orgs_Sim_Mat[tr1,tr2], Dates_Sim_Mat[tr1,tr2], dataText_Sim_Mat[tr1,tr2])).T
        
        dayLabelsList=np.asarray(isdup_labels)
        pred_y=clf.predict(dayinstances)
        conf=clf.predict_proba(dayinstances)
        
        allpred.extend(pred_y)
        alltrue.extend(isdup_labels)
        allconf.extend(conf)
        
        cc=[max(c) for c in conf]
        mask=np.zeros(len(cc),dtype='bool')
#            mm=np.where(np.asarray(cc)<confh)[0]
#            if len(mm)>0:
#                mask[mm]=True
#                countTrainingInstances+=len(mm)
#            else:
#                mask[np.argmin(cc)]=True
#                countTrainingInstances+=1
        
#        Take if among 10% && positve
#        cc1=list(cc)
#        cc1=np.asarray(cc1)
#        minn=int(np.ceil(0.1*len(cc)))
#        mm=cc1.argsort()[:minn]
#        mmm=0
#        while mmm < len(mm):
#            if pred_y[mm[mmm]] == -1:
#                mm=np.delete(mm,mmm,0)
#                mmm-=1
#            mmm+=1
#        
#        mask[mm]=True
#        countTrainingInstances+=len(mm)
        
#        Take min 10% of positive
#        cc1=list(cc)
#        cc1=np.asarray(cc1)[np.where(pred_y==1)[0]]
#        minn=int(np.ceil(0.1*len(cc1)))
#        mm=cc1.argsort()[:minn]
#        for mmm in mm:
#            i=cc.index(cc1[mmm])
#            mask[i]=True
#        
#        countTrainingInstances+=len(mm)
         
#        Take if among 10%
#        cc1=list(cc)
#        cc1=np.asarray(cc1)
#        minn=int(np.ceil(minh*len(cc)))
#        mm=cc1.argsort()[:minn]
#        mask[mm]=True
#        countTrainingInstances+=len(mm)  
        
#        Random 10%
#        minn=int(np.ceil(minh*len(mask)))
#        countTrainingInstances+=minn
#        mask[np.random.randint(0,len(mask),size=minn)]=True
        
#        Update using all
        mask=~mask
        #trainingdata=dayinstances[mask,]
        #traininglabels=dayLabelsList[mask]
        trainingdata=np.vstack((dayinstances[mask,],dayinstances[~mask,]))#
        traininglabels=np.hstack((dayLabelsList[mask],pred_y[~mask]))#
        
        if trainingdata.shape[0]>0:#len(dayindexes)>1:
          clf.partial_fit(trainingdata, traininglabels)
        
        dayindexes=[]



logfilename='results/WindowOfSize_'+str(len(windowinstances))+'_logfile.txt'
print logfilename
#np.where(true_y == pred_y)[0]
print "sklearn.precision_Positive=",100.0*precision_score(alltrue, allpred, pos_label=1, average='binary')
print "sklearn.recall_Positive=",100.0*recall_score(alltrue, allpred, pos_label=1)
print "sklearn.precision_Negative=",100.0*precision_score(alltrue, allpred, pos_label=-1)
print "sklearn.recall_Negative=",100.0*recall_score(alltrue, allpred, pos_label=-1)
print "sklearn.F2=",100.0*fbeta_score(alltrue, allpred, beta=2, average='binary')  #, average='macro'
print "sklearn.F1=",100.0*fbeta_score(alltrue, allpred, beta=1, average='binary')  #, average='macro'
print "sklearn.Accuracy=",100.0*accuracy_score(alltrue, allpred)
print '\n' + logfilename + ' training ratio'  + str(np.unique(LabelsList,return_counts=True))


TP=0
TN=0
FP=0
FN=0
for prd, act in zip(allpred,alltrue):
    if prd == 1 and act==1:
        TP+=1
    elif prd == 1 and act==0:
        FP+=1
    elif prd == 0 and act==0:
        TN+=1
    elif prd == 0 and act==1:
        FN+=1

print "\nTP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN)
#log.append("TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN))

print "\nTPR: " + str(100.0*TP/(TP+FN)) + " TNR: " + str(100.0*TN/(TN+FP)) + " FPR: " + str(100.0*FP/(FP+TN)) + " FNR: " + str(100.0*FN/(FN+TP))
#log.append("TPR: " + str(100.0*TP/(TP+FN)) + " TNR: " + str(100.0*TN/(TN+FP)) + " FPR: " + str(100.0*FP/(FP+TN)) + " FNR: " + str(100.0*FN/(FN+TP)))

import codecs
#f = codecs.open(logfilename,'w',"utf-8")
#f.write("\n".join(log)+"\n") # python will convert \n to os.linesep
##f.write("\n")
#f.close()

print countTrainingInstances, str(TP+TN+FP+FN)
print "\n" + str(minh)


precPos=100.0*precision_score(alltrue, allpred, pos_label=1, average='binary')
recallPos=100.0*recall_score(alltrue, allpred, pos_label=1)
precNeg=100.0*precision_score(alltrue, allpred, pos_label=-1)
recallNeg=100.0*recall_score(alltrue, allpred, pos_label=-1)
f1=100.0*fbeta_score(alltrue, allpred, beta=1, average='binary')
f2=100.0*fbeta_score(alltrue, allpred, beta=2, average='binary')  #, average='macro'
acc=100.0*accuracy_score(alltrue, allpred)

TP=0
TN=0
FP=0
FN=0
for prd, act in zip(allpred,alltrue):
    if prd == 1 and act==1:
        TP+=1
    elif prd == 1 and act==0:
        FP+=1
    elif prd == 0 and act==0:
        TN+=1
    elif prd == 0 and act==1:
        FN+=1

TPR=str(100.0*TP/(TP+FN))
TNR=str(100.0*TN/(TN+FP))
FPR=str(100.0*FP/(FP+TN))
FNR=str(100.0*FN/(FN+TP))

log = [str(nTrainingdays),str(TPR),str(TNR),str(FPR),str(FNR),str(precPos),str(recallPos),str(precNeg),str(recallNeg),str(f1),str(f2),str(acc),"\n"]
print "nTrainingdays,TPR,TNR,FPR,FNR,precPos,recallPos,precNeg,recallNeg,f1,f2,acc,\n"
print log
with codecs.open(logfilename,'a',"utf-8") as f:
    f.write("nTrainingdays,TPR,TNR,FPR,FNR,precPos,recallPos,precNeg,recallNeg,f1,f2,acc,\n" + "\t".join(log))
#print '\nModel size= ' + str(modelMaxsize)
#print '\nTraining size= ' + str(instancesMaxsize)


#import smtplib
# 
#server = smtplib.SMTP('smtp.gmail.com', 587)
#server.starttls()
#server.login("hmdawad3@gmail.com", "")
# 
#msg = "TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN)
#server.sendmail("hmdawad3@gmail.com", "amm106220@utdallas.edu", msg)
#server.quit()

#KeysList.index("NYTFEED020151205ebc5001b9")
#KeysList.index("BBCAP00020151205ebc5001e1")
#AFNWS00020151216ebcg000ed, AFNWS00020151216ebcg0016u, APRS000020151218ebci006h7, AFNWS00020151222ebcm000xv, AFNWS00020151222ebcm000r8
#
#AFPR000020151224ebco004v1, LBA0000020151226ebcq003e9, APRS000020151226ebcq004n0, AFPR000020151227ebcr0035x
#print dataText[KeysList.index("AFPR000020151224ebco004v1")]
#print dataText[KeysList.index("LBA0000020151226ebcq003e9")]
#print dataText[KeysList.index("APRS000020151226ebcq004n0")]
#print dataText[KeysList.index("AFPR000020151227ebcr0035x")]
#AFNWS00020151204ebc400094, AFNWS00020151204ebc4000d4, BBCAP00020151204ebc40012x, AFNWS00020151204ebc4000se, AFNWS00020151204ebc40015v
#print dataText[KeysList.index("AFNWS00020151204ebc400094")]
#print dataText[KeysList.index("AFNWS00020151204ebc4000d4")]
#print dataText[KeysList.index("BBCAP00020151204ebc40012x")]
#print dataText[KeysList.index("AFNWS00020151204ebc4000se")]
#print dataText[KeysList.index("AFNWS00020151204ebc40015v")]
#AFPR000020140218ea2i0061g, BBCMNF0020140218ea2i0050m, BBCMNF0020140218ea2i005xx
#print dataText[KeysList.index("AFPR000020140218ea2i0061g")]
#print dataText[KeysList.index("BBCMNF0020140218ea2i0050m")]
#print dataText[KeysList.index("BBCMNF0020140218ea2i005xx")]
#
#x1=KeysList.index("BBCMNF0020140218ea2i0050m")
#x2=KeysList.index("BBCMNF0020140218ea2i005xx")
#distance.pdist((dataText_features[x1,:],dataText_features[x1,:]), metric='cosine')
#
#AFNWS00020141201eac1001n4, AFPR000020141211eacb006hf 
#x1=KeysList.index("AFNWS00020141201eac1001n4")
#x2=KeysList.index("AFPR000020141211eacb006hf")
#
#print dataText[x1]
#print dataText[x2]
#
#1-distance.pdist((dataText_features[x1,:],dataText_features[x2,:]), metric='cosine')