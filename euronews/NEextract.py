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



directorypath='/home/ahmad/duplicate-detection/euronews/data/jsonfiles2/'
jsondata = [join(directorypath, f) for f in listdir(directorypath) if isfile(join(directorypath, f))]
shuffle(jsondata)

stpwords=set(stopwords.words("spanish")+stopwords.words("english")+stopwords.words("german")+stopwords.words("french"))
alllngpairs=[['en','ar'],['en','fa'],['en','fr'],['en','es'],['en','de'],['ar','fa'],['ar','fr'],['ar','es'],['ar','de'],['fa','fr'],['fa','es'],['fa','de'],['fr','es'],['fr','de'],['es','de']]


stanford_ner_path = '/home/ahmad/nltk_data/stanford/stanford-ner.jar'
os.environ['CLASSPATH'] = stanford_ner_path
stanford_classifier = "/home/ahmad/nltk_data/stanford/es/edu/stanford/nlp/models/ner/spanish.ancora.distsim.s512.crf.ser.gz"
stes = StanfordNERTagger(stanford_classifier)
stanford_classifier = '/home/ahmad/nltk_data/stanford/english.all.3class.distsim.crf.ser.gz'
sten = StanfordNERTagger(stanford_classifier)
stanford_classifier = "/home/ahmad/nltk_data/stanford/de/edu/stanford/nlp/models/ner/german.conll.hgc_175m_600.crf.ser.gz"
stde = StanfordNERTagger(stanford_classifier)


for lngp in alllngpairs[3:4]:
  print lngp
  if lngp[0] in ['ar','fr','fa'] and lngp[1] in ['ar','fr','fa']:
    continue
  
  eventsentspr=[]
  for idx,filenm in enumerate(jsondata):
    with open(filenm,"r") as myfile:
      dayjson=json.load(myfile)
    
    for event in dayjson:
      if any([ll not in event.keys() for ll in lngp]):
          continue
      
      eventsentspr.append([event[lngp[0]]['title'],event[lngp[1]]['title']])
      #lngpair.append(lngp)
  
  
  leftNE=[]
  rightNE=[]
  for pr in eventsentspr:
    timenow=time.time()
    NElist=[]
    if lngp[0] == 'es':
      wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', pr[0])
      classified_text = stes.tag(wordslist)
      NElist=[w[0] for w in classified_text if w[1] not in ['o','O']]
    
    if lngp[0] == 'en':
      wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', pr[0])
      classified_text = sten.tag(wordslist)
      NElist=[w[0] for w in classified_text if w[1] not in ['o','O']]
    
    if lngp[0] == 'de':
      wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', pr[0])
      classified_text = stde.tag(wordslist)
      NElist=[w[0] for w in classified_text if w[1] not in ['o','O']]
    
    leftNE.append(list(set(NElist)))
    
    NElist=[]
    if lngp[1] == 'es':
      wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', pr[1])
      classified_text = stes.tag(wordslist)
      NElist=[w[0] for w in classified_text if w[1] not in ['o','O']]
    
    if lngp[1] == 'en':
      wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', pr[1])
      classified_text = sten.tag(wordslist)
      NElist=[w[0] for w in classified_text if w[1] not in ['o','O']]
    
    if lngp[1] == 'de':
      wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', pr[1])
      classified_text = stde.tag(wordslist)
      NElist=[w[0] for w in classified_text if w[1] not in ['o','O']]
    
    rightNE.append(list(set(NElist)))
    if len(rightNE)%200==0:
      print len(rightNE)
    
    elapsedtime=time.time()-timenow
    break

  
  with open('/home/ahmad/duplicate-detection/euronews/NE/StanfordNEleftEuronews_'+str(lngp).replace("[","").replace("]","").replace(",","").replace("\'","").replace(" ","")+'.txt','w') as myfile:
    myfile.write("\n".join(map(str,leftNE)))
  
  with open('/home/ahmad/duplicate-detection/euronews/NE/StanfordNErightEuronews_'+str(lngp).replace("[","").replace("]","").replace(",","").replace("\'","").replace(" ","")+'.txt','w') as myfile:
    myfile.write("\n".join(map(str,rightNE)))






alllngpairs=[['en','ar'],['en','fa'],['en','fr'],['en','es'],['en','de'],['ar','fa'],['ar','fr'],['ar','es'],['ar','de'],['fa','fr'],['fa','es'],['fa','de'],['fr','es'],['fr','de'],['es','de']]
lngp=['es','de']
leftNE=[]
with open('/home/ahmad/duplicate-detection/euronews/NE/StanfordNEleftEuronews_'+str(lngp[0])+str(lngp[1])+'.txt','r') as myfile:
    leftNE=myfile.readlines()

rightNE=[]
with open('/home/ahmad/duplicate-detection/euronews/NE/StanfordNErightEuronews_'+str(lngp[0])+str(lngp[1])+'.txt','r') as myfile:
    rightNE=myfile.readlines()

len(leftNE),len(rightNE)
leftNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in leftNE]
rightNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in rightNE]
score=[]
for _i,_leftNE,_rightNE in zip(range(len(leftNE)),leftNE,rightNE):
  nintersection=0
  for _iteml in set(_leftNE):
    for _itemr in set(_rightNE):
      if 1.0 * levenshtein(_iteml, _itemr)/(len(_iteml)+len(_itemr))<0.25:
        nintersection+=1
  
  nunion=len(set(_leftNE))+len(set(_rightNE))
  
  if nunion>0:
    score.append(1.0*nintersection/nunion)

npos=len(score)


negativeidx=[]
for _ in range(len(rightNE)):
  negativeidx.append(list(np.random.choice(len(rightNE),2, replace=False))) 


for _i,_pr in enumerate(negativeidx):
  nintersection=0
  _leftNE=leftNE[_pr[0]]
  _rightNE=rightNE[_pr[1]]
  for _iteml in set(_leftNE):
    for _itemr in set(_rightNE):
      if 1.0 * levenshtein(_iteml, _itemr)/(len(_iteml)+len(_itemr))<0.25:
        nintersection+=1
  
  nunion=len(set(_leftNE))+len(set(_rightNE))
  
  if nunion>0:
    score.append(1.0*nintersection/nunion)


def calc(h):
  TP=0
  FP=0
  TN=0
  FN=0
  for _i,_ss in enumerate(score):
    if _ss>=h:
      if _i<npos:
        TP+=1
      elif _i>=npos:
        FP+=1
    elif _ss<h:
      if _i<npos:
        FN+=1
      elif _i>=npos:
        TN+=1
  
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  print Recall



calc(0.5)

h=0.00000001
TP=0
FP=0
TN=0
FN=0
for _i,_ss in enumerate(score):
  if _ss>=h:
    if _i<npos:
      TP+=1
    elif _i>=npos:
      FP+=1
  elif _ss<h:
    if _i<npos:
      FN+=1
    elif _i>=npos:
      TN+=1

Precision=100.0*TP/(TP+FP+0.000001)
Recall=100.0*TP/(TP+FN+0.000001)
F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
print lngp[0],lngp[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))


