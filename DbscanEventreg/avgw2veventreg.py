from sklearn.cross_decomposition import CCA
import time
import sys
import random
from os import listdir
from os.path import isfile, join
import json
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
import cPickle
from multiprocessing import Pool,cpu_count
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from random import shuffle
import re
import numpy as np
from nltk.corpus import stopwords
from fasttext import FastVector
from os import listdir
from os.path import isfile, join
from random import shuffle
import json
from fasttext import FastVector
import re
import numpy as np
from nltk.corpus import stopwords


def loadfasttextmodel(filename):
  filename='/home/ahmad/fastText_multilingual/'
  w2v=dict()
  #['en','es','zh','hr','de','fa','ar','fr']['es','en','de']
  for lng in ['en','es','de','fa','ar','fr']:
    w2v[lng] = FastVector(vector_file=filename+'wiki.'+lng+'.vec')
    w2v[lng].apply_transform(filename+'alignment_matrices/'+lng+'.txt')
  
  return w2v


embeddingsmodel=loadfasttextmodel('Path To Vectors')



stpwords=set(stopwords.words("spanish")+stopwords.words("english")+stopwords.words("german")+stopwords.words("french"))


def create_w2v_pairs(w2vmodel,allposfiles,allnegfiles,lng0,lng1):
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
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['title'])
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
        
        if len(w2vmatrix1)==0:
          continue
        
        sentgroup1.append(artcle['body'])
        lgroup1.append(langcode[artcle['lang']])
        group1.append(np.mean(w2vmatrix1,axis=0))
        wgroup1.append(wlist)
      
      lgroup2=[]
      group2=[]
      wgroup2=[]
      sentgroup2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['title'])
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
        
        if len(w2vmatrix2)==0:
          continue
        
        sentgroup2.append(artcle['body'])
        lgroup2.append(langcode[artcle['lang']])
        group2.append(np.mean(w2vmatrix2,axis=0))
        wgroup2.append(wlist)
      
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          if lgroup1[x1]!=lng0 or lgroup2[x2] != lng1:
            #if lgroup1[x1] not in ['en','es','de'] or lgroup2[x2] not in ['de','es','en']: ,['es','en'],['en','es']['en','de'],['de','en'],['es','en'],['en','es'],['es','de'],['de','es']
            #if lgroup1[x1] not in ['en','es'] or lgroup2[x2] not in ['es','en']:
            continue
          
          pospairs.append([group1[x1],group2[x2]])
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
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['title'])
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
        
        if len(w2vmatrix1)==0:
          continue
        
        sentgroup1.append(artcle['body'])
        lgroup1.append(langcode[artcle['lang']])
        group1.append(np.mean(w2vmatrix1,axis=0))
        wgroup1.append(wlist)
      
      
      
      lgroup2=[]
      group2=[]
      wgroup2=[]
      sentgroup2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['title'])
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
        
        if len(w2vmatrix2)==0:
          continue
        
        sentgroup2.append(artcle['body'])
        lgroup2.append(langcode[artcle['lang']])
        group2.append(np.mean(w2vmatrix2,axis=0))
        wgroup2.append(wlist)
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          if lgroup1[x1]!=lng0 or lgroup2[x2] != lng1:
            #if lgroup1[x1] not in ['en','es','de'] or lgroup2[x2] not in ['de','es','en']:,['es','en'],['en','es'],['es','de'],['de','es']
            #if lgroup1[x1] not in ['en','es'] or lgroup2[x2] not in ['es','en']:
            continue
          
          negpairs.append([group1[x1],group2[x2]])
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



#labels,langpairs,sentleft,sentright=create_w2v_pairs(allposfiles,allnegfiles,lng0=lngp[0],lng1=lngp[1])

posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
allposfiles = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
allnegfiles = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]
to=min(len(allposfiles),len(allnegfiles))


lngp=['de','es']
for lngp in [['es','en'],['de','en'],['de','es']]:
Alllangpairs=[]
Allisdup_labels=[]
embeddingspairs=[]
for frm in range(0,to-50,50):
  w2vpairs,labels, wpairs, langpairs, EntJaccardSim,sentpairs=create_w2v_pairs(embeddingsmodel,allposfiles[frm:frm+50],allnegfiles[frm:frm+50],lng0=lngp[0],lng1=lngp[1])
  if len(labels)==0:
    continue
  
  #print "processing ",frm,len(labels), " pairs"
  embeddingspairs.extend(w2vpairs)
  Allisdup_labels.extend(labels)
  Alllangpairs.extend(langpairs)


sim=[]
for pr in embeddingspairs:
  sim.append(euclidean_distances(pr[0].reshape(1,-1),pr[1].reshape(1,-1)))

ms=max(sim)
normsim=[_sim/ms for _sim in sim]

#euclidean_distances
h=0.605
#TP=sum([True for _s,_lbl in zip(sim,Allisdup_labels) if _s >=h if _lbl==1])
#FN=sum([True for _s,_lbl in zip(sim,Allisdup_labels) if _s < h if _lbl==1])
#TN=sum([True for _s,_lbl in zip(sim,Allisdup_labels) if _s < h if _lbl==0])
#FP=sum([True for _s,_lbl in zip(sim,Allisdup_labels) if _s >=h if _lbl==0])

h=0.39783
#TP=sum([True for _s,_lbl in zip(normsim,Allisdup_labels) if _s >=h if _lbl==1])
#FN=sum([True for _s,_lbl in zip(normsim,Allisdup_labels) if _s < h if _lbl==1])
#TN=sum([True for _s,_lbl in zip(normsim,Allisdup_labels) if _s < h if _lbl==0])
#FP=sum([True for _s,_lbl in zip(normsim,Allisdup_labels) if _s >=h if _lbl==0])

h=0.39183
h=0.5
TP=sum([True for _s,_lbl in zip(normsim,Allisdup_labels) if _s < h if _lbl==1])
FN=sum([True for _s,_lbl in zip(normsim,Allisdup_labels) if _s >=h if _lbl==1])
TN=sum([True for _s,_lbl in zip(normsim,Allisdup_labels) if _s >=h if _lbl==0])
FP=sum([True for _s,_lbl in zip(normsim,Allisdup_labels) if _s < h if _lbl==0])
Precision=100.0*TP/(TP+FP+0.000001)
Recall=100.0*TP/(TP+FN+0.000001)
F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
print "TP",TP,"TN",TN,"FP",FP,"FN",FN
print("\n Accuracy " + str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "\n F1 " + str(F1)+"\n F2 "+ str(F2)+"\n Precision "+ str(Precision)+"\n Recall "+ str(Recall)+ str("\n TPR ")+ str(100.0*TP/(TP+FN+0.0001))+ str("\n TNR ")+ str(100.0*TN/(TN+FP+0.0001))+ str("\n FPR ")+ str( 100.0*FP/(TN+FP+0.0001))+ str("\n FNR ")+ str( 100.0*FN/(TP+FN+0.0001))+"\n n_pos "+str((TP+FN))+"\n n_neg "+str((TN+FP))+"\n positive ratio "+str((1.0*TP+FN)/(TN+FP)))

print lngp[0],lngp[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))

