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
import numpy
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
from nltk.corpus import gazetteers
from sklearn.cluster import DBSCAN

fname='../atrocitiesdata/d2vmodel.model'
model = gensim.models.Doc2Vec.load(fname)

sentences='../atrocitiesdata/TextList3.txt'
dataText=[]
f = open(sentences, 'r')
for d in f:  
  dataText.append(d.decode('utf-8'))

sentences='../atrocitiesdata/ClusterLabel3.txt'
ClusterLabel=[]
f = open(sentences, 'r')
for d in f:
  ClusterLabel.append(int(d.strip()))

sentences='../atrocitiesdata/LocationsList4.txt'
LocationsList=[]
f = open(sentences, 'r')
for d in f:  
    LocationsList.append(d.decode('utf-8'))

sentences='../atrocitiesdata/PersonsList3.txt'
PersonsList=[]
f = open(sentences, 'r')
for d in f:  
    PersonsList.append(d.decode('utf-8'))

sentences='../atrocitiesdata/OrgsList3.txt'
OrgsList=[]
f = open(sentences, 'r')
for d in f:  
    OrgsList.append(d.decode('utf-8'))

sentences='../atrocitiesdata/DatesList3.txt'
DatesList=[]
f = open(sentences, 'r')
for d in f:  
    DatesList.append(d.decode('utf-8'))

sentences='../atrocitiesdata/VictimList3.txt'
VictimList=[]
f = open(sentences, 'r')
for d in f:
  VictimList.append(d.decode('utf-8'))

sentences='../atrocitiesdata/KeysList4.txt'
KeysList=[]
f = open(sentences, 'r')
for d in f:  
  KeysList.append(d.decode('utf-8').strip())

sentences='../atrocitiesdata/FramenetListExpanded.txt'
#sentences='../atrocitiesdata/FramenetList.txt'
FramenetList=[]
f = open(sentences, 'r')
for d in f:  
  FramenetList.append(d.decode('utf-8').strip())


PersonsList.append(" ")
OrgsList.append(" ")
print len(dataText), len(DatesList), len(LocationsList), len(ClusterLabel), len(PersonsList), len(OrgsList), len(VictimList), len(KeysList), len(FramenetList)

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

#
dataText_vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = 'english', ngram_range=(1, 1))
dataText_features = dataText_vectorizer.fit_transform(FramenetList)
dataText_features = dataText_features.toarray()
dataText_vectorizer.get_feature_names()

from dateutil import parser
#rootdir='D:\\NLP\\Mass dataset\\OrderedDocuments\\'
rootdir="../atrocitiesdata/OrderedDocuments"
dirsList=next(os.walk(rootdir))[1]
for d in range(len(dirsList)):
    dirsList[d]= datetime.strptime(dirsList[d], '%d_%b_%Y')

dirsList.sort()
dirsList=[os.path.join(rootdir, date.strftime(s,'%d_%b_%Y')) for s in dirsList]

n=dataText_features.shape[0]
windowsize=3
#instanceslist=[numpy.zeros((1,5)),numpy.zeros((1,5)),numpy.zeros((1,5))]
instanceslist=[numpy.zeros((1,5))]*windowsize
Labelslist=[[1]]*windowsize

windowinstances=[[0]]*windowsize

#clf = SGDClassifier(random_state=100)
clf = SGDClassifier(loss='log', random_state=123456)
#clf = SGDClassifier()
nTrainingdays=10


for day in range(len(dirsList)):
    if day >= nTrainingdays:
        break
    fs=next(os.walk(dirsList[day]))[2]
    Trainingindexes=[]
    for i, j in enumerate(fs):
        key=j.split('.')[0]
        instanceindex=KeysList.index(key)
        Trainingindexes.append(instanceindex)
        
    



#numpy.random.seed(12)
#numpy.random.seed(1234)
numpy.random.seed(123456)
countTrainingInstances=0
confh=0.95
minh=0.03
allpred=[]
alltrue=[]
allconf=[]
dayindexes=[]
outfile=[]
for day in range(nTrainingdays,len(dirsList)):
    #dayindexes=[]
    fs=next(os.walk(dirsList[day]))[2]
    for i, j in enumerate(fs):
        key=j.split('.')[0]
        if key in KeysList:
            instanceindex=KeysList.index(key)
            dayindexes.append(instanceindex)
    
    if day % 5 == 0: #if len(dayindexes)>1:
        Trmask=numpy.zeros(n,dtype='bool')
        Trmask[dayindexes]=True
        datax=dataText_features[Trmask,:]
        
        clusterslabels=[ClusterLabel[i] for i in dayindexes]
        GTclusterslabels=numpy.unique(clusterslabels,return_inverse=True,return_counts=True)
        dayTrlabels=numpy.asarray(ClusterLabel)[Trmask]
        #kmeans = KMeans(n_clusters=len(GTclusterslabels[0]), random_state=0).fit(datax)
        #db = DBSCAN(eps=2, min_samples=2).fit(datax)

        pred_y=[]        
        for x1 in range(datax.shape[0]):
          if sum(datax[x1,:]==0)==datax.shape[1]:
            continue
          for x2 in range(x1+1,datax.shape[0]):
            if sum(datax[x2,:]==0)==datax.shape[1]:
              continue              
            d = distance.pdist(numpy.vstack((datax[x1,:],datax[x2,:])), metric='cosine')[0]
            if numpy.isnan(d):
              print distance.pdist(numpy.vstack((datax[x1,:],datax[x2,:])), metric='cosine')
              print x1, x2
              print datax.shape
              print Trainingindexes
              print sum(datax[x1,:]==0)#um(numpy.isnan(datax[x1,:]))
              break
            if dayTrlabels[x1]==dayTrlabels[x2]:
                outfile.append(str(d)+","+str(1))                  
            else:
                outfile.append(str(d)+","+str(-1))
            pred_y.append(d)        
        
        dayLabelsList=[]
        for x1 in range(dayTrlabels.shape[0]):
          for x2 in range(x1+1,dayTrlabels.shape[0]): 
              if dayTrlabels[x1]==dayTrlabels[x2]:
                dayLabelsList.append(1)
              else:
                dayLabelsList.append(-1)
        
        allpred.extend(pred_y)
        alltrue.extend(dayLabelsList)
        dayindexes=[]
        
    '''
    dataText_Sim_List=[]
    Location_Sim_List=[]
    Person_Sim_List=[]
    Orgs_Sim_List=[]
    Dates_Sim_List=[]
    Labels_Sim_List=[]
    windowi=numpy.hstack(windowinstances)
    for x1 in dayindexes:
        for x2 in windowi:
            dataText_Sim_List.append(distance.pdist((dataText_features[x1,:],dataText_features[x2,:]), metric='cosine'))
            Location_Sim_List.append(1-distance.pdist((Location_features[x1,:],Location_features[x2,:]), metric='jaccard'))
            Person_Sim_List.append(1-distance.pdist((Person_features[x1,:],Person_features[x2,:]), metric='jaccard'))
            Orgs_Sim_List.append(1-distance.pdist((Orgs_features[x1,:],Orgs_features[x2,:]), metric='jaccard'))
            Dates_Sim_List.append(1-distance.pdist((Dates_features[x1,:],Dates_features[x2,:]), metric='jaccard'))
            if ClusterLabel[x1]==ClusterLabel[x2]:
                Labels_Sim_List.append(1)
            else:
                Labels_Sim_List.append(-1)
    
    mdaysinstances=numpy.hstack((Location_Sim_List,Person_Sim_List,Orgs_Sim_List,Dates_Sim_List,dataText_Sim_List))
    mdaysinstances[numpy.isnan(mdaysinstances)]=0
    pred_y=clf.predict(mdaysinstances)
    conf=clf.predict_proba(mdaysinstances)
    allpred.extend(pred_y)
    alltrue.extend(Labels_Sim_List)
    allconf.extend(conf)
    if trainingdata.shape[0]>0:#len(dayindexes)>1:
        clf.partial_fit(trainingdata, traininglabels)
    
#    clf.partial_fit(mdaysinstances, Labels_Sim_List)
#    clf.partial_fit(numpy.vstack((mdaysinstances,dayinstances)), numpy.hstack((Labels_Sim_List,dayLabelsList)))    
    windowinstances[day%windowsize]=dayindexes
    '''
fname='results/expndframesdist.csv'
with open(fname, 'w') as myfile:
  myfile.write("\n".join(outfile))

'''
logfilename='results/WindowOfSize_'+str(len(windowinstances))+'_logfile.txt'
print logfilename
#numpy.where(true_y == pred_y)[0]
print "sklearn.precision_Positive=",100.0*precision_score(alltrue, allpred, pos_label=1, average='binary')
print "sklearn.recall_Positive=",100.0*recall_score(alltrue, allpred, pos_label=1)
print "sklearn.precision_Negative=",100.0*precision_score(alltrue, allpred, pos_label=-1)
print "sklearn.recall_Negative=",100.0*recall_score(alltrue, allpred, pos_label=-1)
print "sklearn.F2=",100.0*fbeta_score(alltrue, allpred, beta=2, average='binary')  #, average='macro'
print "sklearn.F1=",100.0*fbeta_score(alltrue, allpred, beta=1, average='binary')  #, average='macro'
print "sklearn.Accuracy=",100.0*accuracy_score(alltrue, allpred)
print '\n' + logfilename + ' training ratio'  + str(numpy.unique(LabelsList,return_counts=True))
'''
