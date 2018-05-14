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
  #['en','es','zh','hr','de','fa','ar','fr']
  for lng in ['es','en','de']:
    w2v[lng] = FastVector(vector_file=filename+'wiki.'+lng+'.vec')
    w2v[lng].apply_transform(filename+'alignment_matrices/'+lng+'.txt')
  
  #en_vector = w2v['en']["cat"]
  #es_vector = w2v['es']["gato"]
  #print(FastVector.cosine_similarity(es_vector, en_vector))
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


embeddingsmodel=loadfasttextmodel('Path To Vectors')


directorypath='/home/ahmad/duplicate-detection/euronews/data/jsonfiles2/'
jsondata = [join(directorypath, f) for f in listdir(directorypath) if isfile(join(directorypath, f))]
shuffle(jsondata)
stpwords=set(stopwords.words("spanish")+stopwords.words("english")+stopwords.words("german")+stopwords.words("french"))

embeddingspair=[]
wordspair=[]
lngpair=[]
filenm=jsondata[0]
for idx,filenm in enumerate(jsondata):
  with open(filenm,"r") as myfile:
    dayjson=json.load(myfile)
  
  for event in dayjson:
    if 'en' not in event.keys() or 'ar' not in event.keys():
        continue
    
    eventembeddings=[]
    eventwords=[]
    eventlngs=[]
    for lng in event.keys():
      if lng in ['id','fr','de','pt','gr','hu','ru','tr','it','fa','es']:
        continue
      
      wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', event[lng]['title'])
      docembeddings=[]
      wlist=[]
      for word in wordslist:
        if '' !=word.strip().lower() and word.strip().lower() not in stpwords:
          try:
            if type(word)!=type(''):
              word=word.strip().lower().encode('utf-8')
            else:
              word=word.strip().lower()
            
            #w2vmatrix1.append(w2vmodel[word])
            docembeddings.append(transformedmodelaren[lng][word])
            #docembeddings.append(embeddingsmodel0[lng][word])
            wlist.append(word)
            
          except:
            pass
      
      eventembeddings.append(np.array(docembeddings))
      eventwords.append(wlist)
      eventlngs.append(lng.encode('utf-8'))
    
    embeddingspair.append(eventembeddings)
    wordspair.append(eventwords)
    lngpair.append(eventlngs)


negativeidx=[]
for _ in range(2*len(embeddingspair)):
  negativeidx.append(list(np.random.choice(len(embeddingspair),2, replace=False))) 


