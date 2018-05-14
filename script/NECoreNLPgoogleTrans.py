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


stanford_ner_path = '/home/ahmad/nltk_data/stanford/stanford-ner.jar'
os.environ['CLASSPATH'] = stanford_ner_path
stanford_classifier = "/home/ahmad/nltk_data/stanford/es/edu/stanford/nlp/models/ner/spanish.ancora.distsim.s512.crf.ser.gz"
stes = StanfordNERTagger(stanford_classifier)
stanford_classifier = '/home/ahmad/nltk_data/stanford/english.all.3class.distsim.crf.ser.gz'
sten = StanfordNERTagger(stanford_classifier)
stanford_classifier = "/home/ahmad/nltk_data/stanford/de/edu/stanford/nlp/models/ner/german.conll.hgc_175m_600.crf.ser.gz"
stde = StanfordNERTagger(stanford_classifier)


service = build('translate', 'v2',developerKey='AIzaSyCqpf3hXzheoI9ttfw9JWhMRHtYt5Z72X4')


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
          if lgroup1[x1]!='es' or lgroup2[x2]!='en':
            continue
          possentpairs.append([sentgroup1[x1],sentgroup2[x2]])
          poslangpairs.append([lgroup1[x1],lgroup2[x2]])
          continue
          if [lgroup1[x1],lgroup2[x2]] not in [['de','en'],['es','en'],['de','es']]:
            #if [lgroup1[x1],lgroup2[x2]] not in [['en','de'],['de','en'],['es','en'],['en','es']]:
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
          if lgroup1[x1]!='es' or lgroup2[x2]!='en':
            continue
          neglangpairs.append([lgroup1[x1],lgroup2[x2]])
          negsentpairs.append([sentgroup1[x1],sentgroup2[x2]])
          continue
          if [lgroup1[x1],lgroup2[x2]] not in [['de','en'],['es','en'],['de','es']]:
            #if [lgroup1[x1],lgroup2[x2]] in [['en','de'],['de','en'],['es','en'],['en','es']]:
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



stpwords=set(stopwords.words("spanish")+stopwords.words("english")+stopwords.words("german")+stopwords.words("french"))

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

len(Alllangpairs)
#,len(allrightNE)
'''
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
'''


a=[]
n=0
score=[]
allleftNE=[]
allrightNE=[]
transallleftNE=[]
transallrightNE=[]
lbl=[]
lng=[]
for _c in range(17):
  leftNE=[]
  with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordNEleftAttro2enesde_'+str(_c)+'.txt','r') as myfile:
    leftNE=myfile.readlines()
  
  rightNE=[]
  with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordNErightAttro2enesde_'+str(_c)+'.txt','r') as myfile:
    rightNE=myfile.readlines()
  
  leftNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in leftNE]
  rightNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in rightNE]
  
  allleftNE.extend(leftNE)
  allrightNE.extend(rightNE)

for _i,_leftNE,_rightNE in zip(range(len(allleftNE)),allleftNE,allrightNE):
  if _i % 1000==0:
    print _i, len(transallleftNE)
  
  try:
    if ((Alllangpairs[_i][0]=='es' or Alllangpairs[_i][0]=='de') and Alllangpairs[_i][1]=='en') or (Alllangpairs[_i][0]=='de' and Alllangpairs[_i][1]=='es'):
      
      translation=service.translations().list(source=Alllangpairs[_i][0],target='en',q=_leftNE,format='text').execute()
      transwordsl=[transw['translatedText'].encode('utf-8') for transw in translation['translations']]
      
      
      if Alllangpairs[_i][1]!='en':
        translation=service.translations().list(source=Alllangpairs[_i][1],target='en',q=_rightNE,format='text').execute()
        transwords=[transw['translatedText'].encode('utf-8') for transw in translation['translations']]
      else:
        transwords=_rightNE
      
      
      
      nunion=len(set(transwordsl+transwords))
      nintersection=0
      for _iteml in set(transwordsl):
        if _iteml in transwords:
          nintersection += 1
      
      transallleftNE.append(transwordsl)
      transallrightNE.append(transwords)
      if nunion>0:
        score.append(1.0*nintersection/nunion)
      else:
        score.append(-1)
      
      lbl.append(Allisdup_labels[_i])
      lng.append(Alllangpairs[_i])
    #else:
    #  transallleftNE.append(_leftNE)
    #  transallrightNE.append(_rightNE)
  except:
    #transallleftNE.append('')
    #transallrightNE.append('')
    a.append(_i)
    pass


##----------STOP
with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordNEleftgoogleTransenesde.txt', 'wb') as myfile:
      myfile.write("\n".join(map(str,transallleftNE))) #list of lists

with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordNErightgoogleTransenesde.txt', 'wb') as myfile:
      myfile.write("\n".join(map(str,transallrightNE)))

with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/lblStanfordNEgoogleTransenesde.txt', 'wb') as myfile:
      myfile.write("\n".join(lbl))

lng0=[_lng[0] for _lng in lng] 
with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/leftStanfordNElnggoogleTransenesde.txt', 'wb') as myfile:
      myfile.write("\n".join(lng0))

lng1=[_lng[1] for _lng in lng]
with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/rightStanfordNElnggoogleTransenesde.txt', 'wb') as myfile:
      myfile.write("\n".join(lng1))

with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/scoreStanfordNEgoogleTransenesde.txt', 'wb') as myfile:
      myfile.write("\n".join(map(str,score)))

len(Alllangpairs),len(score),len(Allsentpairs),len(Allisdup_labels)

h=.08
TP=FP=FN=TN=0
for _score,_lbl,_lng in zip(score,lbl,lng):
  if _lng[0]!='de' or _lng[1]!='es':
    continue
  
  if _score>=h and _lbl==1:
    TP+=1
  elif _score>=h and _lbl==0:
    FP+=1
  elif _score<h and _lbl==0:
    TN+=1
  elif _score<h and _lbl==1:
    FN+=1

Precision=100.0*TP/(TP+FP)
Recall=100.0*TP/(TP+FN)
F1=100.0*(2.0*TP)/(2.0*TP+1.0*FN+FP)
F2=100.0*(5.0*TP)/(5.0*TP+4.0*FN+FP)
print("TP",TP,"TN",TN,"FP",FP,"FN",FN)
print TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))

print Precision,Recall
print 100.0*FP/(FP+TN),100.0*FN/(FN+TP)

TP+FP+FN+TN



#Stats
countpos=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==1 and _lng[0]=='es' and _lng[1]=='en'])
countneg=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==0 and _lng[0]=='es' and _lng[1]=='en'])
print countpos,countneg
TP=31
TN=621
FP=25+countneg
FN=91+countpos
Precision=100.0*TP/(TP+FP)
Recall=100.0*TP/(TP+FN)
F1=100.0*(2.0*TP)/(2.0*TP+1.0*FN+FP)
F2=100.0*(5.0*TP)/(5.0*TP+4.0*FN+FP)
print("TP",TP,"TN",TN,"FP",FP,"FN",FN)
print TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))

print Precision,Recall
print 100.0*FP/(FP+TN),100.0*FN/(FN+TP)

countpos=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==1 and _lng[0]=='de' and _lng[1]=='en'])
countneg=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==0 and _lng[0]=='de' and _lng[1]=='en'])
print countpos,countneg
TP=9315
TN=115
FP=countneg
FN=54+countpos
Precision=100.0*TP/(TP+FP)
Recall=100.0*TP/(TP+FN)
F1=100.0*(2.0*TP)/(2.0*TP+1.0*FN+FP)
F2=100.0*(5.0*TP)/(5.0*TP+4.0*FN+FP)
print("TP",TP,"TN",TN,"FP",FP,"FN",FN)
print TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))

print Precision,Recall
print 100.0*FP/(FP+TN),100.0*FN/(FN+TP)
countpos=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==1 and _lng[0]=='de' and _lng[1]=='es'])
countneg=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==0 and _lng[0]=='de' and _lng[1]=='es'])
print countpos,countneg

TP=5
TN=45
FP=countneg
FN=countpos
Precision=100.0*TP/(TP+FP)
Recall=100.0*TP/(TP+FN)
F1=100.0*(2.0*TP)/(2.0*TP+1.0*FN+FP)
F2=100.0*(5.0*TP)/(5.0*TP+4.0*FN+FP)
print("TP",TP,"TN",TN,"FP",FP,"FN",FN)
print TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))

print Precision,Recall
print 100.0*FP/(FP+TN),100.0*FN/(FN+TP)

################
lngscore=[]
lnglbl=[]
_lng0='de'
_lng1='es'
for idx in range(len(transallleftNE)):
  if lng[idx][0]==_lng0 and lng[idx][1]==_lng1:
    transwordsl=transallleftNE[idx]
    transwords=transallrightNE[idx]
    nunion=len(set(transwordsl+transwords))
    nintersection=0
    for _iteml in set(transwordsl):
      if _iteml in transwords:
        nintersection += 1
    
    lnglbl.append(lbl[idx])
    if nunion>0 and len(transwordsl)>0 and len(transwords)>0:
      lngscore.append(1.0*nintersection/nunion)
    else:
      lngscore.append(-1)

len(lnglbl),len(lngscore)

countpos=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==1 and _lng[0]==_lng0 and _lng[1]==_lng1])
countneg=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==0 and _lng[0]==_lng0 and _lng[1]==_lng1])

h=.07
TP=FP=FN=TN=0
for _score,_lbl in zip(lngscore,lnglbl):
  if _score==-1 and _lbl==0:
    #FP+=1
    continue
  
  if _score==-1 and _lbl==1:
    #FN+=1
    continue
  
  if _score>=h and _lbl==1:
    TP+=1
  elif _score>=h and _lbl==0:
    FP+=1
  elif _score<h and _lbl==0:
    TN+=1
  elif _score<h and _lbl==1:
    FN+=1

d=countpos-(TP+FN)
FN+=d
d=countneg-(TN+FP)
FP+=d

Precision=100.0*TP/(TP+FP)
Recall=100.0*TP/(TP+FN)
F1=100.0*(2.0*TP)/(2.0*TP+1.0*FN+FP)
F2=100.0*(5.0*TP)/(5.0*TP+4.0*FN+FP)
print("TP",TP,"TN",TN,"FP",FP,"FN",FN)
print TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))

print Precision,Recall
print 100.0*FP/(FP+TN),100.0*FN/(FN+TP)

TP+FP+FN+TN