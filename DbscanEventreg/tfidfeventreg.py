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



stpwords=set(stopwords.words("spanish")+stopwords.words("english")+stopwords.words("german")+stopwords.words("french"))


def create_w2v_pairs(allposfiles,allnegfiles,lng0='en',lng1='es'):
  langcode={"eng":"en","spa":"es","deu":"de","zho":"zh","ita":"it","fra":"fr","rus":"ru","swe":"sv","nld":"nl","tur":"tr","jpn":"ja","por":"pt","ara":"ar","fin":"fi","ron":"ro","kor":"ko","hrv":"hr","tam":"","hun":"hu","slv":"sl","pol":"pl","srp":"sr","cat":"ca","ukr":"uk"}
  #w2vmodel= loadmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
  possentleft=[]
  possentright=[]
  langpairs=[]
  labels = []
  sentpairsleft=[]
  sentpairsright=[]
  poslangpairs=[]
  #print("creating positive pairs:")
  for idx,Pfilenm in enumerate(allposfiles):
    try:
      with open(Pfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      lgroup1=[]
      sentgroup1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        sentgroup1.append(artcle['title'])
        lgroup1.append(langcode[artcle['lang']])
      
      lgroup2=[]
      sentgroup2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        sentgroup2.append(artcle['title'])
        lgroup2.append(langcode[artcle['lang']])
      
      
      for x1 in range(len(lgroup1)):
        for x2 in range(len(lgroup2)):
          if lgroup1[x1]!=lng0 or lgroup2[x2]!=lng1:
            #if [lgroup1[x1],lgroup2[x2]] not in [['en','de'],['de','en'],['es','en'],['en','es'],['es','de'],['de','es']]:
            #if lgroup1[x1] not in ['en','es','de'] or lgroup2[x2] not in ['de','es','en']: ,['es','en'],['en','es']
            #if lgroup1[x1] not in ['en','es'] or lgroup2[x2] not in ['es','en']:
            continue
          
          poslangpairs.append([lgroup1[x1],lgroup2[x2]])
          #possentpairs.append([sentgroup1[x1],sentgroup2[x2]])
          possentleft.append(sentgroup1[x1])
          possentright.append(sentgroup2[x2])
      
      
      
      #sys.stdout.write("\r")
      #sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allposfiles)), 100*idx/len(allposfiles)))
      #sys.stdout.flush()
      
    except:
      pass
  
  
  #print('\ncreating negative pairs...')
  
  neglangpairs=[]
  negsentleft=[]
  negsentright=[]
  for idx,Nfilenm in enumerate(allnegfiles):
    try:
      with open(Nfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      
      lgroup1=[]
      sentgroup1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        sentgroup1.append(artcle['title'])
        lgroup1.append(langcode[artcle['lang']])
      
      
      
      lgroup2=[]
      sentgroup2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        sentgroup2.append(artcle['title'])
        lgroup2.append(langcode[artcle['lang']])
      
      for x1 in range(len(lgroup1)):
        for x2 in range(len(lgroup2)):
          if lgroup1[x1]!=lng0 or lgroup2[x2]!=lng1:
            #if [lgroup1[x1],lgroup2[x2]] not in [['en','de'],['de','en'],['es','en'],['en','es'],['es','de'],['de','es']]:
            #if lgroup1[x1] not in ['en','es','de'] or lgroup2[x2] not in ['de','es','en']:
            #if lgroup1[x1] not in ['en','es'] or lgroup2[x2] not in ['es','en']:
            continue
          
          neglangpairs.append([lgroup1[x1],lgroup2[x2]])
          #negsentpairs.append([sentgroup1[x1],sentgroup2[x2]])
          negsentleft.append(sentgroup1[x1])
          negsentright.append(sentgroup2[x2])
      
      
      #sys.stdout.write("\r")
      #sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allnegfiles)), 100*idx/len(allnegfiles)))
      #sys.stdout.flush()
    except:
      pass
  
  
  #print("\nShuffling...")
  #print len(possentleft),len(possentright),len(poslangpairs),len(negsentleft),len(negsentright),len(neglangpairs)
  for posspairl,posspairr,poslpair in zip(possentleft,possentright,poslangpairs):
    labels.append(1)
    langpairs.append(poslpair)
    sentpairsleft.append(posspairl)
    sentpairsright.append(posspairr)
  
  for negspairl,negspairr,neglpair in zip(negsentleft,negsentright,neglangpairs):
    labels.append(0)
    langpairs.append(neglpair)
    sentpairsleft.append(negspairl)
    sentpairsright.append(negspairr)
  
  
  return labels, langpairs,sentpairsleft,sentpairsright

#labels,langpairs,sentleft,sentright=create_w2v_pairs(allposfiles,allnegfiles,lng0=lngp[0],lng1=lngp[1])

posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
allposfiles = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
allnegfiles = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]
to=min(len(allposfiles),len(allnegfiles))


for lngp in [['en','es'],['es','en'],['de','en'],['en','de'],['es','de'],['de','es']]:
  Alllangpairs=[]
  Allisdup_labels=[]
  Allsentleft=[]
  Allsentright=[]
  for frm in range(0,to-50,50):
    labels,langpairs,sentleft,sentright=create_w2v_pairs(allposfiles[frm:frm+50],allnegfiles[frm:frm+50],lng0=lngp[0],lng1=lngp[1])
    if len(labels)==0:
      continue
    
    #print "processing ",frm,len(labels), " pairs"
    Allisdup_labels.extend(labels)
    Alllangpairs.extend(langpairs)
    Allsentleft.extend(sentleft)
    Allsentright.extend(sentright)
  
  
  dataText_vectorizer1 = CountVectorizer(analyzer = "word", stop_words=stpwords, binary=True) #,ngram_range=(1, 1), max_features = 1000
  dataText_features1 = dataText_vectorizer1.fit_transform(Allsentleft)
  dataText_features1 = dataText_features1.toarray()
  
  dataText_vectorizer2 = CountVectorizer(analyzer = "word", stop_words=stpwords, binary=True) #,ngram_range=(1, 1), max_features = 1000
  dataText_features2 = dataText_vectorizer2.fit_transform(Allsentright)
  dataText_features2 = dataText_features2.toarray()
  
  #cca = CCA(n_components=min(dataText_features1.shape[1],dataText_features2.shape[1])/5)
  cca = CCA(n_components=10)
  
  pos1=[dt for dt,_lbl in zip(dataText_features1,Allisdup_labels) if _lbl==1]
  pos2=[dt for dt,_lbl in zip(dataText_features2,Allisdup_labels) if _lbl==1]
  n_tr=min(1000,len(pos1)/3)
  train1=np.array(pos1[:n_tr])
  train2=np.array(pos2[:n_tr])
  cca.fit(train1, train2)
  neg1=[dt for dt,_lbl in zip(dataText_features1,Allisdup_labels) if _lbl==0]
  neg2=[dt for dt,_lbl in zip(dataText_features2,Allisdup_labels) if _lbl==0]
  X_l, X_r = cca.transform(neg1, neg2)
  sim=[]
  for _X_l, _X_r in zip(X_l, X_r):
    sim.append(cosine_similarity(_X_l.reshape(1,-1),_X_r.reshape(1,-1)))
  
  FP=sum([True for _s in sim if _s >=0.5])
  TN=sum([True for _s in sim if _s < 0.5])
  
  X_l, X_r = cca.transform(pos1[n_tr:], pos2[n_tr:])
  sim=[]
  for _X_l, _X_r in zip(X_l, X_r):
    sim.append(cosine_similarity(_X_l.reshape(1,-1),_X_r.reshape(1,-1)))
  
  TP=sum([True for _s in sim if _s >=0.5])
  FN=sum([True for _s in sim if _s < 0.5])
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
  F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
  
  print lngp[0],lngp[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))



#print("\n Accuracy " + str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "\n F1 " + str(F1)+"\n F2 "+ str(F2)+"\n Precision "+ str(Precision)+"\n Recall "+ str(Recall)+ str("\n TPR ")+ str(100.0*TP/(TP+FN+0.0001))+ str("\n TNR ")+ str(100.0*TN/(TN+FP+0.0001))+ str("\n FPR ")+ str( 100.0*FP/(TN+FP+0.0001))+ str("\n FNR ")+ str( 100.0*FN/(TP+FN+0.0001))+"\n n_pos "+str((TP+FN))+"\n n_neg "+str((TN+FP))+"\n positive ratio "+str((1.0*TP+FN)/(TN+FP+0.0001)))

lngp=['de','es']
for lngp in [['en','es'],['es','en'],['de','en'],['en','de'],['es','de'],['de','es']]:
  Alllangpairs=[]
  Allisdup_labels=[]
  Allsentleft=[]
  Allsentright=[]
  for frm in range(0,to-50,50):
    labels,langpairs,sentleft,sentright=create_w2v_pairs(allposfiles[frm:frm+50],allnegfiles[frm:frm+50],lng0=lngp[0],lng1=lngp[1])
    if len(labels)==0:
      continue
    
    #print "processing ",frm,len(labels), " pairs"
    Allisdup_labels.extend(labels)
    Alllangpairs.extend(langpairs)
    Allsentleft.extend(sentleft)
    Allsentright.extend(sentright)
  
  
  dataText_vectorizer1 = TfidfVectorizer(analyzer = "word", stop_words=stpwords) #,ngram_range=(1, 1), max_features = 1000
  dataText_features1 = dataText_vectorizer1.fit_transform(Allsentleft)
  dataText_features1 = dataText_features1.toarray()
  
  dataText_vectorizer2 = TfidfVectorizer(analyzer = "word", stop_words=stpwords) #,ngram_range=(1, 1), max_features = 1000
  dataText_features2 = dataText_vectorizer2.fit_transform(Allsentright)
  dataText_features2 = dataText_features2.toarray()
  
  #cca = CCA(n_components=min(dataText_features1.shape[1],dataText_features2.shape[1])/5)
  cca = CCA(n_components=10)
  X_l, X_r = cca.fit_transform(dataText_features1[:-1], dataText_features2[:-1])
  sim=[]
  for _X_l, _X_r in zip(X_l, X_r):
    sim.append(cosine_similarity(_X_l.reshape(1,-1),_X_r.reshape(1,-1)))
  
  FP=sum([True for _s,_lbl in zip(sim,Allisdup_labels) if _s >=0.5 if _lbl==0])
  TN=sum([True for _s,_lbl in zip(sim,Allisdup_labels) if _s < 0.5 if _lbl==0])
  TP=sum([True for _s,_lbl in zip(sim,Allisdup_labels) if _s >=0.5 if _lbl==1])
  FN=sum([True for _s,_lbl in zip(sim,Allisdup_labels) if _s < 0.5 if _lbl==1])
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
  F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
  
  print lngp[0],lngp[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))






