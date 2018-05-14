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


def loadw2vmodel(filename):
  w2v=dict()
  #filename='fifty_nine.table5.multiCluster.m_1000+iter_10+window_3+min_count_5+size_40.normalized'
  #filename='/home/ahmad/duplicate-detection/multilingual-embedding/three.table4.multiSkip.iter_10+window_3+min_count_5+size_40.normalized'
  with open(filename, "r") as myfile:
    for line in myfile:
      lineparts=line.strip().split(":")
      #if lineparts[0] not in ['fr','en','es', 'zh', 'hr', 'de']:
      if lineparts[0] not in ['en','es','de']:
        continue
      wordvector=lineparts[1].split(" ")
      if lineparts[0] not in w2v.keys():
        w2v[lineparts[0]]=dict()
      
      if type(wordvector[0])==type(''):
        #w2v[wordvector[0].decode('utf-8')]=list(map(float,wordvector[1:]))
        w2v[lineparts[0]][wordvector[0].decode('utf-8')]=map(float,wordvector[1:])
      else:
        #w2v[wordvector[0]]=list(map(float,wordvector[1:]))
        w2v[lineparts[0]][wordvector[0]]=map(float,wordvector[1:])
  
  return w2v


stanford_ner_path = '/home/ahmad/nltk_data/stanford/stanford-ner.jar'
os.environ['CLASSPATH'] = stanford_ner_path
stanford_classifier = "/home/ahmad/nltk_data/stanford/es/edu/stanford/nlp/models/ner/spanish.ancora.distsim.s512.crf.ser.gz"
stes = StanfordNERTagger(stanford_classifier)
stanford_classifier = '/home/ahmad/nltk_data/stanford/english.all.3class.distsim.crf.ser.gz'
sten = StanfordNERTagger(stanford_classifier)
stanford_classifier = "/home/ahmad/nltk_data/stanford/de/edu/stanford/nlp/models/ner/german.conll.hgc_175m_600.crf.ser.gz"
stde = StanfordNERTagger(stanford_classifier)

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
        sentgroup1.append(artcle['body'])
        lgroup1.append(langcode[artcle['lang']])
        continue
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
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
        sentgroup2.append(artcle['body'])
        lgroup2.append(langcode[artcle['lang']])
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
      
      
      for x1 in range(len(lgroup1)):
        for x2 in range(len(lgroup2)):
          possentpairs.append([sentgroup1[x1],sentgroup2[x2]])
          poslangpairs.append([lgroup1[x1],lgroup2[x2]])
          continue
          if [lgroup1[x1],lgroup2[x2]] in [['en','de'],['de','en'],['es','en'],['en','es']]:
            #if lgroup1[x1] not in ['en','es','de'] or lgroup2[x2] not in ['de','es','en']:
            #if lgroup1[x1] not in ['en','es'] or lgroup2[x2] not in ['es','en']:
            continue
          
          pospairs.append([np.array(group1[x1]),np.array(group2[x2])])
          poswordpairs.append([wgroup1[x1],wgroup2[x2]])
          poslangpairs.append([lgroup1[x1],lgroup2[x2]])
          posEntJaccardSim.append(jsonfile['meta']['entityJaccardSim'])
          possentpairs.append([sentgroup1[x1],sentgroup2[x2]])
      
      
      
      #sys.stdout.write("\r")
      #sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allposfiles)), 100*idx/len(allposfiles)))
      #sys.stdout.flush()
      
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
        sentgroup1.append(artcle['body'])
        lgroup1.append(langcode[artcle['lang']])
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
        lgroup2.append(langcode[artcle['lang']])
        sentgroup2.append(artcle['body'])
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
      
      for x1 in range(len(lgroup1)):
        for x2 in range(len(lgroup2)):
          neglangpairs.append([lgroup1[x1],lgroup2[x2]])
          negsentpairs.append([sentgroup1[x1],sentgroup2[x2]])
          continue
          if [lgroup1[x1],lgroup2[x2]] in [['en','de'],['de','en'],['es','en'],['en','es']]:
            #if lgroup1[x1] not in ['en','es','de'] or lgroup2[x2] not in ['de','es','en']:
            #if lgroup1[x1] not in ['en','es'] or lgroup2[x2] not in ['es','en']:
            continue
          
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
  #print len(pospairs),len(negpairs),len(poswordpairs),len(negwordpairs),
  #print len(poslangpairs),len(neglangpairs),len(possentpairs),len(negsentpairs)
  for posspair,poslpair in zip(possentpairs,poslangpairs):
    labels.append(1)
    langpairs.append(poslpair)
    sentpairs.append(posspair)
  
  for negspair,neglpair in zip(negsentpairs,neglangpairs):
    labels.append(0)
    langpairs.append(neglpair)
    sentpairs.append(negspair)
  
  return labels, langpairs,sentpairs


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

unifiedw2vmodel=loadw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')

unifiedw2vmodel=dict()
AllEntJaccardSim=[]
Allwordspairs=[]
Allsentpairs=[]
w2vpairsList=[]
Alllangpairs=[]
Allisdup_labels=[]
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
  labels,langpairs,sentpairs=create_w2v_pairs(unifiedw2vmodel,[posfilenames[frm]],[negfilenames[frm]])
  if len(labels)==0:
    continue
  
  #print "processing ",frm,len(w2vpairs),len(w2vpairs[0]), " pairs"
  if frm%50 == 0 and frm>0:
    print frm
    
  Allisdup_labels.extend(labels)
  Alllangpairs.extend(langpairs)
  Allsentpairs.extend(sentpairs)


print len(Allsentpairs),len(Allisdup_labels),len(Alllangpairs)
print len(rightNE),len(leftNE)
#must clean if lng[0] not in ['de','en','es'] and lng[1] not in ['de','en','es']:
rightNE[i]=[]
leftNE[i]=[]
leftNE=[]
rightNE=[]
for i,lng in enumerate(Alllangpairs):
  if lng[0] not in ['de','en','es'] and lng[1] not in ['de','en','es']:
    break

for i,lng in zip(range(min(len(leftNE),len(rightNE))),Alllangpairs):
  if lng[0] not in ['de','en','es']:
    leftNE[i]=[]
  
  if lng[1] not in ['de','en','es']:
    rightNE[i]=[]



for i,artcl,lbl,lng in zip(range(len(Allsentpairs)),Allsentpairs,Allisdup_labels,Alllangpairs):
  if i<n:
    continue
  
n=2795
n=5000
n=2891
n=4000
n=5086
n=2927
n=50000
n=100000
n=540000

i=2795
n=2790
leftNE=[]
rightNE=[]
for i,artcl,lbl,lng in zip(range(len(Allsentpairs)),Allsentpairs,Allisdup_labels,Alllangpairs):
  if i==4000:
    break
  
  
  if i<n:
    NElist=[]
    if i<len(leftNE):
      leftNE[i]=NElist
    else:
      leftNE.append(NElist)
    
    if i<len(rightNE):
      rightNE[i]=NElist
    else:
      rightNE.append(NElist)
    continue
  
  if lng[0] not in ['de','en','es'] or lng[1] not in ['de','en','es']:
    NElist=[]
    if i<len(leftNE):
      leftNE[i]=NElist
    else:
      leftNE.append(NElist)
    
    if i<len(rightNE):
      rightNE[i]=NElist
    else:
      rightNE.append(NElist)
    continue
  
  NElist=[]
  if lng[0] == 'es':
    wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcl[0])
    classified_text = stes.tag(wordslist)
    NElist=[w[0] for w in classified_text if w[1] not in ['o','O']]
  
  if lng[0] == 'en':
    wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcl[0])
    classified_text = sten.tag(wordslist)
    NElist=[w[0] for w in classified_text if w[1] not in ['o','O']]
  
  if lng[0] == 'de':
    wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcl[0])
    classified_text = stde.tag(wordslist)
    NElist=[w[0] for w in classified_text if w[1] not in ['o','O']]
  
  if i<len(leftNE):
    leftNE[i]=list(set(NElist))
  else:
    leftNE.append(list(set(NElist)))
  
  NElist=[]
  if lng[1] == 'es':
    wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcl[1])
    classified_text = stes.tag(wordslist)
    NElist=[w[0] for w in classified_text if w[1] not in ['o','O']]
  
  if lng[1] == 'en':
    wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcl[1])
    classified_text = sten.tag(wordslist)
    NElist=[w[0] for w in classified_text if w[1] not in ['o','O']]
  
  if lng[1] == 'de':
    wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcl[1])
    classified_text = stde.tag(wordslist)
    NElist=[w[0] for w in classified_text if w[1] not in ['o','O']]
  
  if i<len(rightNE):
    rightNE[i]=list(set(NElist))
  else:
    rightNE.append(list(set(NElist)))
  
  if i%1000==0:
    print i

print len(leftNE), len(rightNE)


with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordNEleftAttro2enesde_14.txt','w') as myfile:
  myfile.write("\n".join(map(str,rightNE[500000:540000])))

with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordNErightAttro2enesde_14.txt','w') as myfile:
  myfile.write("\n".join(map(str,leftNE[500000:540000])))

print len(leftNE), len(rightNE)

"_0" [:4000] DONE
"_1" [4000:5000] DONE
"_2" [5000:50000] DONE 
"_3" [50000:100000] DONE
"_4" [100000:150000] DONE
"_5" [150000:200000] DONE
"_6" [200000:250000] DONE
"_7" [250000:300000] DONE
"_8" [300000:335388] DONE
"_9" [335388:350000] DONE
"_10" [350000:356714] DONE
"_11" [356714:400000] DONE
"_12" [400000:450000] DONE
"_13" [450000:500000] DONE
"_14" [500000:540000] DONE
"_15" [540000:550000] DONE
"_16" [550000:600000] DONE


n=0
score=[]
for _c in range(17):
  leftNE=[]
  with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordNEleftAttro2enesde_'+str(_c)+'.txt','r') as myfile:
    leftNE=myfile.readlines()
  
  rightNE=[]
  with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordNErightAttro2enesde_'+str(_c)+'.txt','r') as myfile:
    rightNE=myfile.readlines()
  
  leftNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in leftNE]
  rightNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in rightNE]
  for _i,_leftNE,_rightNE in zip(range(len(leftNE)),leftNE,rightNE):
    if Alllangpairs[n+_i][0]==Alllangpairs[n+_i][1]:
      score.append(-1)
      continue
    
    nintersection=0
    for _iteml in set(_leftNE):
      for _itemr in set(_rightNE):
        if 1.0 * levenshtein(_iteml, _itemr)/(len(_iteml)+len(_itemr))<0.25:
          nintersection+=1
    
    nunion=len(set(_leftNE))+len(set(_rightNE))
    
    if nunion>0:
      score.append(1.0*nintersection/nunion)
  
  n+=len(leftNE)


len(Alllangpairs),len(score),len(Allsentpairs),len(Allisdup_labels)
i=score.index(-1)
sum([True for _s, _l in zip(score, Alllangpairs[540000:540000+len(score)]) if _l[0]==_l[1] and _s==-1])
sum([True for _s in score if _s==-1])

def calc(lg,h):
  TP=0
  FP=0
  TN=0
  FN=0
  for _ss,_lbl,_lng in zip(score,Allisdup_labels,Alllangpairs):
    if _lng[0]==lg[0] and _lng[1]==lg[1]:
      if _ss==-1:
        if _lbl==0:
          FP+=1
        elif _lbl==1:
          FN+=1
        
        continue
      
      if _ss>=h:
        if _lbl==1:
          TP+=1
        elif _lbl==0:
          FP+=1
      elif _ss<h:
        if _lbl==1:
          FN+=1
        elif _lbl==0:
          TN+=1
  
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  print Recall

  F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
  F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
  print("\n Accuracy " + str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "\n F1 " + str(F1)+"\n F2 "+ str(F2)+"\n Precision "+ str(Precision)+"\n Recall "+ str(Recall)+ str("\n TPR ")+ str(100.0*TP/(TP+FN+0.0001))+ str("\n TNR ")+ str(100.0*TN/(TN+FP+0.0001))+ str("\n FPR ")+ str( 100.0*FP/(TN+FP+0.0001))+ str("\n FNR ")+ str( 100.0*FN/(TP+FN+0.0001))+"\n n_pos "+str((TP+FN))+"\n n_neg "+str((TN+FP))+"\n positive ratio "+str((1.0*TP+FN)/(TN+FP)))

lngp=['de','es']
calc(lngp,0.5)

h=0.00000001
TP=0
FP=0
TN=0
FN=0
for _ss,_lbl,_lng in zip(score,Allisdup_labels,Alllangpairs):
  if _lng[0]==lngp[0] and _lng[1]==lngp[1]:
    if _ss==-1:
      if _lbl==0:
          FP+=1
      elif _lbl==1:
        FN+=1
      
      continue
    
    if _ss>=h:
      if _lbl==1:
        TP+=1
      elif _lbl==0:
        FP+=1
    elif _ss<h:
      if _lbl==1:
        FN+=1
      elif _lbl==0:
        TN+=1

Precision=100.0*TP/(TP+FP+0.000001)
Recall=100.0*TP/(TP+FN+0.000001)
F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
print lngp[0],lngp[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))



n=0
score=[]
AllleftNE=[]
AllrightNE=[]
for _c in range(17):
  leftNE=[]
  with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordNEleftAttro2enesde_'+str(_c)+'.txt','r') as myfile:
    leftNE=myfile.readlines()
  
  rightNE=[]
  with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordNErightAttro2enesde_'+str(_c)+'.txt','r') as myfile:
    rightNE=myfile.readlines()
  
  leftNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in leftNE]
  rightNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in rightNE]
  AllleftNE.extend(leftNE)
  AllrightNE.extend(rightNE)
  n+=len(leftNE)


len(Alllangpairs),len(AllleftNE),len(AllrightNE),len(Allsentpairs),len(Allisdup_labels)

machingleftNE=[]
machingrightNE=[]
for lngpr,leftNE,rightNE in zip(Alllangpairs,AllleftNE,AllrightNE):
  if lngpr[0]=='en' and lngpr[1]=='es':
    for _iteml in set(leftNE):
      for _itemr in set(rightNE):
        if 1.0 * levenshtein(_iteml, _itemr)/(len(_iteml)+len(_itemr))<0.25:
          machingleftNE.append(_iteml)
          machingrightNE.append(_itemr)
    
  
  elif lngpr[1]=='es' and lngpr[0]=='en':
    for _iteml in set(leftNE):
      for _itemr in set(rightNE):
        if 1.0 * levenshtein(_iteml, _itemr)/(len(_iteml)+len(_itemr))<0.25:
          machingleftNE.append(_itemr)
          machingrightNE.append(_iteml)

len(machingleftNE),len(machingrightNE)


with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNEleftAttro2enes.txt','w') as myfile:
  myfile.write("\n".join(map(str,machingleftNE)))

with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNErightAttro2enes.txt','w') as myfile:
  myfile.write("\n".join(map(str,machingrightNE)))




Allsentpairs[540000+i]
Allisdup_labels[540000+i]
Alllangpairs[540000+i]

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



print len(Allsentpairs),len(Allisdup_labels),len(Alllangpairs)



for artcle in jsonfile[keys[0]]['articles']['results']:
  sentences=re.split(r'[,.?]', artcle['body'])
  wlist=[]
  for sent in sentences:
    wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', sent)
    if langcode[artcle['lang']] == 'es':
      classified_text = stes.tag(wordslist)
      NEs=[w[0] for w in classified_text if w[1] not in ['o','O']]
    if langcode[artcle['lang']] == 'en':
      classified_text = sten.tag(wordslist)
      NEs=[w[0] for w in classified_text if w[1] not in ['o','O']]
    else:
      NEs=[]
    
    NEs=set(NEs)
    if len(NEs)==1:
      wlist.append(NEs)
    elif len(NEs)>1:
      wlist.extend(NEs)

data = ["ACCTCCTAGAAG", "ACCTACTAGAAGTT", "GAATATTAGGCCGA"]
def lev_metric(x, y):
    i, j = int(x[0]), int(y[0])     # extract indices
    return levenshtein(data[i], data[j])

levenshtein("ACCTCCTAGAAG", "ACCTACTAGAAGTT")

"ACCTCCTAGAAG", 
"ACCTACTAGAAGTT"

dataText1=[a+' ' +b+' ' +c for a,b,c in zip(Allloclists,Allorgslist,Allpeoplelist)]
dataText2=[a+' ' +b+' ' +c for a,b,c in zip(Allloclists2,Allorgslist2,Allpeoplelist2)]

numclusters=[]
clustersdist=[]
pureclustersratio=[]
for pidx in range(len(dataText1)):
  dt1=dataText1[pidx].split(" ")
  dt1 = filter(None, dt1)
  dt1= [dt.decode('utf-8') if type(dt)==type('') else dt for dt in dt1]
  dt2=dataText2[pidx].split(" ")
  dt2 = filter(None, dt2)
  dt2= [dt.decode('utf-8') if type(dt)==type('') else dt for dt in dt2]
  if len(dt1) ==0 or len(dt2) ==0:
    numclusters.append([-100000,-100000,-100000])
    pureclustersratio.append(-100000)
    clustersdist.append([])
    continue
  data=dt1+dt2
  X = np.arange(len(data)).reshape(-1, 1)
  Y=[1]*len(dt1)+[2]*len(dt2)
  labels_=dbscan(X, metric=lev_metric, eps=2, min_samples=4)
  if len(labels_)!=2:
    numclusters.append([-100000,-100000,-100000])
    pureclustersratio.append(-100000)
    clustersdist.append([])
    continue
  ll=np.unique(labels_[1])
  n_pure_cl=0
  n_noise_cl=0
  n_mixed_cl=0
  clusters=dict()
  for _idx,_lbl in enumerate(labels_[1]):
    if _lbl==-1:
      continue
    if _lbl not in clusters.keys():
      clusters[_lbl]=[]
    
    clusters[_lbl].append(Y[_idx])
  
  for _lbl in clusters.keys():
    if _lbl == -1:
      n_noise_cl+=1
    elif len(set(clusters[_lbl]))>1:
      n_mixed_cl+=1
    else:
      n_pure_cl+=1
  
  clustersdist.append(clusters)#pairclusters)
  numclusters.append([n_pure_cl,n_mixed_cl,n_noise_cl])
  pureclustersratio.append(1.0*n_pure_cl/(n_pure_cl+n_mixed_cl+0.00001))

return numclusters, pureclustersratio, clustersdist

sum([True if pp[0]==0 and pp[1]==1 else False for pp in numclusters])
sum([True if pp[2]>0 else False for pp in numclusters])

def clustering_purity(_w2vpairs,dbscan_eps=0.5, dbscan_minPts=5):
  numclusters=[]
  clustersdist=[]
  pureclustersratio=[]
  #_w2vpairs=w2vpairs
  #print len(_w2vpairs)
  for pridx in range(len(_w2vpairs)):
    #start = time.time()
    if _w2vpairs[pridx][0].size ==0 or _w2vpairs[pridx][1].size ==0:
      numclusters.append([-100000,-100000,-100000])
      pureclustersratio.append(-100000)
      clustersdist.append([])
      continue
    
    X=np.vstack((_w2vpairs[pridx][0],_w2vpairs[pridx][1]))
    Y=[1]*_w2vpairs[pridx][0].shape[0]+[2]*_w2vpairs[pridx][1].shape[0]
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, n_jobs=-1).fit(X)
    ll=np.unique(db.labels_)
    n_pure_cl=0
    n_noise_cl=0
    n_mixed_cl=0
    clusters=dict()
    for _idx,_lbl in enumerate(db.labels_):
      if _lbl==-1:
        continue
      if _lbl not in clusters.keys():
        clusters[_lbl]=[]
      
      clusters[_lbl].append(Y[_idx])
    
    for _lbl in clusters.keys():
      if _lbl == -1:
        n_noise_cl+=1
      elif len(set(clusters[_lbl]))>1:
        n_mixed_cl+=1
      else:
        n_pure_cl+=1
    
    
    #print( time.time() - start)
    clustersdist.append(clusters)#pairclusters)
    numclusters.append([n_pure_cl,n_mixed_cl,n_noise_cl])
    pureclustersratio.append(1.0*n_pure_cl/(n_pure_cl+n_mixed_cl+0.00001))
  
  return numclusters, pureclustersratio, clustersdist
  
  
