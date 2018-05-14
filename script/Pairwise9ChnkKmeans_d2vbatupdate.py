# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:29:20 2016

@author: Ahmad
"""

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
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
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
import gensim
from datetime import datetime,date,timedelta
#import load
import numpy
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

#fname='../atrocitiesdata/d2vmodel.model'
#model = gensim.models.Doc2Vec.load(fname)

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

PersonsList.append(" ")
OrgsList.append(" ")
print len(dataText), len(DatesList), len(LocationsList), len(ClusterLabel), len(PersonsList), len(OrgsList), len(VictimList), len(KeysList)

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


'''
def get_doc_list(sentencesfile):
  dataText=[]
  f = open(sentencesfile, 'r')
  for d in f:
    dataText.append(d.decode('utf-8'))
  
  return dataText

def get_doc(sentencesfile):
  doc_list = get_doc_list(sentencesfile)
  tokenizer = RegexpTokenizer(r'\w+')
  #en_stop = get_stop_words('en')
  en_stop = set(stopwords.words('english'))
  p_stemmer = PorterStemmer() 
  taggeddoc = []
  texts = []
  for index,i in enumerate(doc_list):
      # for tagged doc
      wordslist = []
      tagslist = []
      # clean and tokenize document string
      raw = i.lower()
      tokens = tokenizer.tokenize(raw)
      # remove stop words from tokens
      stopped_tokens = [i for i in tokens if not i in en_stop]
      # remove numbers
      number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
      number_tokens = ' '.join(number_tokens).split()
      # stem tokens
      stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
      # remove empty
      length_tokens = [i for i in stemmed_tokens if len(i) > 1]
      # add tokens to list
      texts.append(length_tokens)
      td = TaggedDocument((' '.join(stemmed_tokens)).split(),[str(index)])
      #td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(),str(index))
      taggeddoc.append(td)
  
  return taggeddoc
'''

def sents_to_taggeddoc(doc_list):
  #doc_list = get_doc_list(sentencesfile)
  tokenizer = RegexpTokenizer(r'\w+')
  #en_stop = get_stop_words('en')
  en_stop = set(stopwords.words('english'))
  p_stemmer = PorterStemmer() 
  taggeddoc = []
  texts = []
  for index,i in enumerate(doc_list):
      # for tagged doc
      wordslist = []
      tagslist = []
      # clean and tokenize document string
      raw = i.lower()
      tokens = tokenizer.tokenize(raw)
      # remove stop words from tokens
      stopped_tokens = [i for i in tokens if not i in en_stop]
      # remove numbers
      number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
      number_tokens = ' '.join(number_tokens).split()
      # stem tokens
      stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
      # remove empty
      length_tokens = [i for i in stemmed_tokens if len(i) > 1]
      # add tokens to list
      texts.append(length_tokens)
      #td = TaggedDocument((' '.join(stemmed_tokens)).split(),str(index))
      td = TaggedDocument((' '.join(stemmed_tokens)).split(),[str(index)])
      #td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(),str(index))
      taggeddoc.append(td)
  
  return taggeddoc

def trainmodel(documentsbatch):
  model = gensim.models.Doc2Vec(documentsbatch, size= 20)
  for epoch in range(50):
      if epoch % 40 == 0:
          print ('Now training epoch %s'%epoch)
      
      model.train(documentsbatch)
  
  return model


def updatemodel(documentsbatch,model):
  for epoch in range(50):
      #if epoch % 40 == 0:
      #    print ('Now training epoch %s'%epoch)
      
      model.train(documentsbatch)
  
  return model

#


'''
dataText_vectorizer = TfidfVectorizer(analyzer = "word",tokenizer = tokenize, preprocessor = None, stop_words = 'english', max_features = 10000) #, ngram_range=(1, 1)
dataText_features = dataText_vectorizer.fit_transform(dataText)
dataText_features = dataText_features.toarray()

#d = distance.pdist(dataText_features[Trmask,:], metric='cosine')
#dataText_dist_Mat = distance.squareform(d)
#dataText_Sim_Mat=dataText_dist_Mat
#dataText_Sim_Mat[numpy.isnan(dataText_Sim_Mat)]=0

Location_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, binary=True, preprocessor = None, stop_words = None, max_features = 10000) #, ngram_range=(1, 1)
Location_features = Location_vectorizer.fit_transform(LocationsList)
Location_features = Location_features.toarray()
#d = distance.pdist(Location_features[Trmask,:], metric='jaccard')
#Location_dist_Mat = distance.squareform(d)
#Location_Sim_Mat=1-Location_dist_Mat
#Location_Sim_Mat[numpy.isnan(Location_Sim_Mat)]=0

Person_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, binary=True, preprocessor = None, stop_words = None, max_features = 10000) #, ngram_range=(1, 1)
Person_features = Person_vectorizer.fit_transform(PersonsList)
Person_features = Person_features.toarray()
#d = distance.pdist(Person_features[Trmask,:], metric='jaccard')
#Person_dist_Mat = distance.squareform(d)
#Person_Sim_Mat=1-Person_dist_Mat
#Person_Sim_Mat[numpy.isnan(Person_Sim_Mat)]=0

Orgs_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, preprocessor = None, binary=True, stop_words = None, max_features = 10000) #, ngram_range=(1, 1)
Orgs_features = Orgs_vectorizer.fit_transform(OrgsList)
Orgs_features = Orgs_features.toarray()
#d = distance.pdist(Orgs_features[Trmask,:], metric='jaccard')
#Orgs_dist_Mat = distance.squareform(d)
#Orgs_Sim_Mat=1-Orgs_dist_Mat
#Orgs_Sim_Mat[numpy.isnan(Orgs_Sim_Mat)]=0

Dates_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, preprocessor = None, binary=True, stop_words = None, max_features = 10000) #, ngram_range=(1, 1)
Dates_features = Dates_vectorizer.fit_transform(DatesList)
Dates_features = Dates_features.toarray()
#d = distance.pdist(Dates_features[Trmask,:], metric='jaccard')
#Dates_dist_Mat = distance.squareform(d)
#Dates_Sim_Mat=1-Dates_dist_Mat
#Dates_Sim_Mat[numpy.isnan(Dates_Sim_Mat)]=0
'''

from dateutil import parser
#rootdir='D:\\NLP\\Mass dataset\\OrderedDocuments\\'
rootdir="../atrocitiesdata/OrderedDocuments"
dirsList=next(os.walk(rootdir))[1]
for d in range(len(dirsList)):
    dirsList[d]= datetime.strptime(dirsList[d], '%d_%b_%Y')

dirsList.sort()
dirsList=[os.path.join(rootdir, date.strftime(s,'%d_%b_%Y')) for s in dirsList]

n=len(dataText)
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
        
    '''
    Trmask=numpy.zeros(n,dtype='bool')
    Trmask[Trainingindexes]=True
    
    d = distance.pdist(dataText_features[Trmask,:], metric='cosine')
    dataText_dist_Mat = distance.squareform(d)
    dataText_Sim_Mat=dataText_dist_Mat
    dataText_Sim_Mat[numpy.isnan(dataText_Sim_Mat)]=0
    
    d = distance.pdist(Location_features[Trmask,:], metric='jaccard')
    Location_dist_Mat = distance.squareform(d)
    Location_Sim_Mat=1-Location_dist_Mat
    Location_Sim_Mat[numpy.isnan(Location_Sim_Mat)]=0
    
    d = distance.pdist(Person_features[Trmask,:], metric='jaccard')
    Person_dist_Mat = distance.squareform(d)
    Person_Sim_Mat=1-Person_dist_Mat
    Person_Sim_Mat[numpy.isnan(Person_Sim_Mat)]=0
    
    d = distance.pdist(Orgs_features[Trmask,:], metric='jaccard')
    Orgs_dist_Mat = distance.squareform(d)
    Orgs_Sim_Mat=1-Orgs_dist_Mat
    Orgs_Sim_Mat[numpy.isnan(Orgs_Sim_Mat)]=0
    
#    Dates_features
    
#    if dirsList[day].split("/")[1].lower() in Dates_vectorizer.get_feature_names():
#        Dates_features[Trmask,Dates_vectorizer.get_feature_names().index(dirsList[day].split("/")[1].lower())]=0    
    d = distance.pdist(Dates_features[Trmask,:], metric='jaccard')
    Dates_dist_Mat = distance.squareform(d)
    Dates_Sim_Mat=1-Dates_dist_Mat
    Dates_Sim_Mat[numpy.isnan(Dates_Sim_Mat)]=0
    
    Trlabels=numpy.asarray(ClusterLabel)[Trmask]
    TrPairwiseLabel=numpy.zeros((Trlabels.shape[0],Trlabels.shape[0]))
    for x1 in range(TrPairwiseLabel.shape[0]):
        for x2 in range(x1+1,TrPairwiseLabel.shape[0]): 
            if Trlabels[x1]==Trlabels[x2]:
                TrPairwiseLabel[x1,x2]=1 #Related pair
                TrPairwiseLabel[x2,x1]=5 # 5 = ignore lower triangler matrix entries
            else:
                TrPairwiseLabel[x1,x2]=-1 #Unrelated pair
                TrPairwiseLabel[x2,x1]=-5
    
    tr1=[]
    tr2=[]
    for i1 in range(TrPairwiseLabel.shape[0]):
        tr1.extend(numpy.repeat(i1,(TrPairwiseLabel.shape[0]-i1-1)))
        tr2.extend(range(i1+1,TrPairwiseLabel.shape[0]))
    
    dayinstances=numpy.vstack((Location_Sim_Mat[tr1,tr2], Person_Sim_Mat[tr1,tr2], Orgs_Sim_Mat[tr1,tr2], Dates_Sim_Mat[tr1,tr2], dataText_Sim_Mat[tr1,tr2])).T
    dayLabelsList=TrPairwiseLabel[tr1,tr2]
    instanceslist[day%windowsize]=dayinstances
    Labelslist[day%windowsize]=dayLabelsList
    
    windowinstances[day%windowsize]=Trainingindexes
    

instances=numpy.vstack(instanceslist)
LabelsList=[item for sublist in Labelslist for item in sublist]
trinstances=instances
trLabelsList=LabelsList
#clf.fit(trinstances, trLabelsList)
clf.partial_fit(trinstances, trLabelsList,numpy.unique(trLabelsList))
'''


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
model = None
for day in range(nTrainingdays,len(dirsList)):
    #dayindexes=[]
    fs=next(os.walk(dirsList[day]))[2]
    for i, j in enumerate(fs):
        key=j.split('.')[0]
        if key in KeysList:
            instanceindex=KeysList.index(key)
            dayindexes.append(instanceindex)
    
    if day % 5 == 0: #if len(dayindexes)>1:
        documents =sents_to_taggeddoc([dataText[i] for i in dayindexes])
        if model is None :
          model=trainmodel(documents)
        else:
          model=updatemodel(documents,model)
        
        data=[]
        for i in range(len(documents)):
          data.append(model.infer_vector(documents[i][0]))
        
        datax=numpy.asarray(data)
        
        Trmask=numpy.zeros(n,dtype='bool')
        Trmask[dayindexes]=True
        
        dayTrlabels=numpy.asarray(ClusterLabel)[Trmask]
        
        #clusterslabels=[ClusterLabel[i] for i in Trainingindexes]
        GTclusterslabels=numpy.unique(dayTrlabels)#,return_inverse=True,return_counts=True)
        
        #kmeans = KMeans(n_clusters=len(GTclusterslabels), random_state=0).fit(datax)
        db = DBSCAN(eps=0.25, min_samples=2).fit(datax)
        pred_y=[]
        for x1 in range(db.labels_.shape[0]):
          for x2 in range(x1+1,db.labels_.shape[0]):
              if db.labels_[x1]!= -1 and db.labels_[x1]==db.labels_[x2]:
                  pred_y.append(1)
              else:
                  pred_y.append(-1)
        
        #dayTrlabels=numpy.asarray(ClusterLabel)[Trmask]
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
        d = distance.pdist(dataText_features[Trmask,:], metric='cosine')
        dataText_dist_Mat = distance.squareform(d)
        dataText_Sim_Mat=dataText_dist_Mat
        dataText_Sim_Mat[numpy.isnan(dataText_Sim_Mat)]=0
        
        d = distance.pdist(Location_features[Trmask,:], metric='jaccard')
        Location_dist_Mat = distance.squareform(d)
        Location_Sim_Mat=1-Location_dist_Mat
        Location_Sim_Mat[numpy.isnan(Location_Sim_Mat)]=0
        
        d = distance.pdist(Person_features[Trmask,:], metric='jaccard')
        Person_dist_Mat = distance.squareform(d)
        Person_Sim_Mat=1-Person_dist_Mat
        Person_Sim_Mat[numpy.isnan(Person_Sim_Mat)]=0
        
        d = distance.pdist(Orgs_features[Trmask,:], metric='jaccard')
        Orgs_dist_Mat = distance.squareform(d)
        Orgs_Sim_Mat=1-Orgs_dist_Mat
        Orgs_Sim_Mat[numpy.isnan(Orgs_Sim_Mat)]=0
        
        d = distance.pdist(Dates_features[Trmask,:], metric='jaccard')
        Dates_dist_Mat = distance.squareform(d)
        Dates_Sim_Mat=1-Dates_dist_Mat
        Dates_Sim_Mat[numpy.isnan(Dates_Sim_Mat)]=0
        
        dayTrlabels=numpy.asarray(ClusterLabel)[Trmask]
        TrPairwiseLabel=numpy.zeros((dayTrlabels.shape[0],dayTrlabels.shape[0]))
        for x1 in range(TrPairwiseLabel.shape[0]):
          for x2 in range(x1+1,TrPairwiseLabel.shape[0]): 
              if dayTrlabels[x1]==dayTrlabels[x2]:
                  TrPairwiseLabel[x1,x2]=1 #Duplicate pair
                  TrPairwiseLabel[x2,x1]=5 # 5 = ignore lower triangler matrix entries
              else:
                  TrPairwiseLabel[x1,x2]=-1 #Non-duplicate pair
                  TrPairwiseLabel[x2,x1]=-5
        
        tr1=[]
        tr2=[]
        for i1 in range(TrPairwiseLabel.shape[0]):
            tr1.extend(numpy.repeat(i1,(TrPairwiseLabel.shape[0]-i1-1)))
            tr2.extend(range(i1+1,TrPairwiseLabel.shape[0]))
        
        dayLabelsList=TrPairwiseLabel[tr1,tr2]        
        dayinstances=numpy.vstack((Location_Sim_Mat[tr1,tr2], Person_Sim_Mat[tr1,tr2], Orgs_Sim_Mat[tr1,tr2], Dates_Sim_Mat[tr1,tr2], dataText_Sim_Mat[tr1,tr2])).T
        pred_y=clf.predict(dayinstances)
        conf=clf.predict_proba(dayinstances)
        
        
        
        
        allpred.extend(pred_y)
        alltrue.extend(dayLabelsList)
        allconf.extend(conf)
        
        cc=[max(c) for c in conf]
        mask=numpy.zeros(len(cc),dtype='bool')
#            mm=numpy.where(numpy.asarray(cc)<confh)[0]
#            if len(mm)>0:
#                mask[mm]=True
#                countTrainingInstances+=len(mm)
#            else:
#                mask[numpy.argmin(cc)]=True
#                countTrainingInstances+=1
        
#        Take if among 10% && positve
#        cc1=list(cc)
#        cc1=numpy.asarray(cc1)
#        minn=int(numpy.ceil(0.1*len(cc)))
#        mm=cc1.argsort()[:minn]
#        mmm=0
#        while mmm < len(mm):
#            if pred_y[mm[mmm]] == -1:
#                mm=numpy.delete(mm,mmm,0)
#                mmm-=1
#            mmm+=1
#        
#        mask[mm]=True
#        countTrainingInstances+=len(mm)
        
#        Take min 10% of positive
#        cc1=list(cc)
#        cc1=numpy.asarray(cc1)[numpy.where(pred_y==1)[0]]
#        minn=int(numpy.ceil(0.1*len(cc1)))
#        mm=cc1.argsort()[:minn]
#        for mmm in mm:
#            i=cc.index(cc1[mmm])
#            mask[i]=True
#        
#        countTrainingInstances+=len(mm)

#        Take if among 10%
#        cc1=list(cc)
#        cc1=numpy.asarray(cc1)
#        minn=int(numpy.ceil(minh*len(cc)))
#        mm=cc1.argsort()[:minn]
#        mask[mm]=True
#        countTrainingInstances+=len(mm)  
        
#        Random 10%
#        minn=int(numpy.ceil(minh*len(mask)))
#        countTrainingInstances+=minn
#        mask[numpy.random.randint(0,len(mask),size=minn)]=True
        
#        Update using all
        mask=~mask
        #trainingdata=dayinstances[mask,]
        #traininglabels=dayLabelsList[mask]
        trainingdata=numpy.vstack((dayinstances[mask,],dayinstances[~mask,]))#
        traininglabels=numpy.hstack((dayLabelsList[mask],pred_y[~mask]))#
        
        if trainingdata.shape[0]>0:#len(dayindexes)>1:
          clf.partial_fit(trainingdata, traininglabels)
        
        
    
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
#print '\n' + logfilename + ' training ratio'  + str(numpy.unique(LabelsList,return_counts=True))


TP=0
TN=0
FP=0
FN=0
for prd, act in zip(allpred,alltrue):
    if prd == 1 and act==1:
        TP+=1
    elif prd == 1 and act==-1:
        FP+=1
    elif prd == -1 and act==-1:
        TN+=1
    elif prd == -1 and act==1:
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
    elif prd == 1 and act==-1:
        FP+=1
    elif prd == -1 and act==-1:
        TN+=1
    elif prd == -1 and act==1:
        FN+=1

TPR=str(100.0*TP/(TP+FN))
TNR=str(100.0*TN/(TN+FP))
FPR=str(100.0*FP/(FP+TN))
FNR=str(100.0*FN/(FN+TP))

log = [str(nTrainingdays),str(TPR),str(TNR),str(FPR),str(FNR),str(precPos),str(recallPos),str(precNeg),str(recallNeg),str(f1),str(f2),str(acc),"\n"]
print "nTrainingdays,TPR,TNR,FPR,FNR,precPos,recallPos,precNeg,recallNeg,f1,f2,acc,\n"
print log
with codecs.open(logfilename,'a',"utf-8") as f:
    f.write("d2vupKmeans,nTrainingdays,TPR,TNR,FPR,FNR,precPos,recallPos,precNeg,recallNeg,f1,f2,acc,\n" + "\t".join(log))
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