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
  for lng in ['en','es','de']:
    w2v[lng] = FastVector(vector_file=filename+'wiki.'+lng+'.vec')
    w2v[lng].apply_transform(filename+'alignment_matrices/'+lng+'.txt')
  
  return w2v


embeddingsmodel=loadfasttextmodel('Path To Vectors')


directorypath='/home/ahmad/duplicate-detection/euronews/data/jsonfiles2/'
jsondata = [join(directorypath, f) for f in listdir(directorypath) if isfile(join(directorypath, f))]
shuffle(jsondata)

stpwords=set(stopwords.words("spanish")+stopwords.words("english")+stopwords.words("german")+stopwords.words("french"))
alllngpairs=[['en','ar'],['en','fa'],['en','fr'],['en','es'],['en','de'],['ar','fa'],['ar','fr'],['ar','es'],['ar','de'],['fa','fr'],['fa','es'],['fa','de'],['fr','es'],['fr','de'],['es','de']]

print "EuroNews Avg w2v cos results..."
lngp = alllngpairs[3]
#for lngp in alllngpairs:
embeddingspair=[]

for idx,filenm in enumerate(jsondata):
  with open(filenm,"r") as myfile:
    dayjson=json.load(myfile)
  
  for event in dayjson:
    if any([ll not in event.keys() for ll in lngp]):
        continue
    
    timenow=time.time()
    #eventsentspr.append([event[lngp[0]]['title'],event[lngp[1]]['title']])
    #lngpair.append(lngp)
    wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', event[lngp[0]]['title'])
    docembeddings1=[]
    
    for word in wordslist:
      if '' !=word.strip().lower() and word.strip().lower() not in stpwords:
        try:
          if type(word)!=type(''):
            word=word.strip().lower().encode('utf-8')
          else:
            word=word.strip().lower()
          
          docembeddings1.append(embeddingsmodel[lngp[0]][word])
          
        except:
          pass
    
    wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', event[lngp[1]]['title'])
    docembeddings2=[]
    for word in wordslist:
      if '' !=word.strip().lower() and word.strip().lower() not in stpwords:
        try:
          if type(word)!=type(''):
            word=word.strip().lower().encode('utf-8')
          else:
            word=word.strip().lower()
          
          docembeddings2.append(embeddingsmodel[lngp[1]][word])
          
        except:
          pass
    
    if len(docembeddings1)==0 or len(docembeddings2)==0:
      continue
    
    
    embeddingspair.append([np.mean(docembeddings1,axis=0),np.mean(docembeddings2,axis=0)])
    if len(embeddingspair)>0:
      elapsedtime=time.time()-timenow
      break
    
  if len(embeddingspair)>0:
    break


negativeidx=[]
for _ in range(len(embeddingspair)):
  negativeidx.append(list(np.random.choice(len(embeddingspair),2, replace=False))) 

import sklearn.metrics.pairwise.euclidean_distances
#cosine_similarity
sim=[]
for pr in embeddingspair:
  sim.append(sklearn.metrics.pairwise.euclidean_distances(pr[0].reshape(1,-1),pr[1].reshape(1,-1)))

timenow=time.time()
sklearn.metrics.pairwise.euclidean_distances(pr[0].reshape(1,-1),pr[1].reshape(1,-1))
elapsedtime=time.time()-timenow


npos=len(sim)
for pr in negativeidx:
  sim.append(euclidean_distances(embeddingspair[pr[0]][0].reshape(1,-1),embeddingspair[pr[1]][1].reshape(1,-1)))

ms=max(sim)
normsim=[_sim/ms for _sim in sim]

h=0.5313845
calc(0.496)
def calc(h):
  TP=sum([True for _s in normsim[:npos] if _s < h])
  FN=sum([True for _s in normsim[:npos] if _s >=h])
  FP=sum([True for _s in normsim[npos:] if _s < h])
  TN=sum([True for _s in normsim[npos:] if _s >=h])
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
  F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
  print Recall

h=0.5041845
TP=sum([True for _s in normsim[:npos] if _s < h])
FN=sum([True for _s in normsim[:npos] if _s >=h])
FP=sum([True for _s in normsim[npos:] if _s < h])
TN=sum([True for _s in normsim[npos:] if _s >=h])
Precision=100.0*TP/(TP+FP+0.000001)
Recall=100.0*TP/(TP+FN+0.000001)
F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
print lngp[0],lngp[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))

#print "TP",TP,"TN",TN,"FP",FP,"FN",FN
#print("\n Accuracy " + str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "\n F1 " + str(F1)+"\n F2 "+ str(F2)+"\n Precision "+ str(Precision)+"\n Recall "+ str(Recall)+ str("\n TPR ")+ str(100.0*TP/(TP+FN+0.0001))+ str("\n TNR ")+ str(100.0*TN/(TN+FP+0.0001))+ str("\n FPR ")+ str( 100.0*FP/(TN+FP+0.0001))+ str("\n FNR ")+ str( 100.0*FN/(TP+FN+0.0001))+"\n n_pos "+str((TP+FN))+"\n n_neg "+str((TN+FP))+"\n positive ratio "+str((1.0*TP+FN)/(TN+FP)))


