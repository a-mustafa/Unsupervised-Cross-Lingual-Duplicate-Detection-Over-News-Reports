from leven import levenshtein
import cPickle
#import gzip
import os
import sys
import time
import scipy.io

from keras.models import Model,Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape,Activation
from keras.optimizers import Adam,SGD

#from keras.regularizers import activity_l1
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from keras import losses
from keras.layers.advanced_activations import LeakyReLU, PReLU
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from nltk import StanfordNERTagger
from fasttext import FastVector
from google.cloud import translate
from googleapiclient.discovery import build

def loadfasttextmodel(filename):
  filename='/home/ahmad/fastText_multilingual/'
  w2v=dict()
  #['en','es','zh','hr','de','fa','ar','fr']['es','en','de']
  for lng in ['en','es']:
    w2v[lng] = FastVector(vector_file=filename+'wiki.'+lng+'.vec')
    #w2v[lng].apply_transform(filename+'alignment_matrices/'+lng+'.txt')
  
  return w2v



def loadtransfasttextmodel(filename):
  filename='/home/ahmad/fastText_multilingual/'
  w2v=dict()
  #['en','es','zh','hr','de','fa','ar','fr']['es','en','de']
  for lng in ['en','es','ar']:
    w2v[lng] = FastVector(vector_file=filename+'wiki.'+lng+'.vec')
    w2v[lng].apply_transform(filename+'alignment_matrices/'+lng+'.txt')
  
  return w2v

'''
enwords=[]
lng = 'en'
for word in embeddingsmodel0[lng].id2word:
      if len(word)>1:
        enwords.append(word)

with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/FastTextenwords.txt', 'wb') as myfile:
      myfile.write("\n".join(enwords))

print len(enwords)


service = build('translate', 'v2',developerKey='AIzaSyCqpf3hXzheoI9ttfw9JWhMRHtYt5Z72X4')
arwordsTrans=[]

n=len(arwordsTrans)
for idx,word in enumerate(enwords[:10000]):
      if idx < n:
        continue
      if idx % 1000==0:
        print idx, len(arwordsTrans)
      
      try:
        url = 'http://api.geonames.org/search?q='+word+'&maxRows=1&username='+username[idx%len(username)]
        response = urllib2.urlopen(url)
        webContent = response.read()
        y=BeautifulSoup()
        web=BeautifulSoup(webContent, features="lxml")
        if int(web.geonames.totalresultscount.text)>0:
          translation=service.translations().list(source='en',target='ar',q=[word],format='text').execute()
          arwordsTrans.append(word + " : " + translation['translations'][0]['translatedText'].encode('utf-8'))
        else:
          arwordsTrans.append('')
      
      except:
        arwordsTrans.append('')
        pass

print len(arwordsTrans)
for word in arwordsTrans[5000:]:
  if word!='':
    arwordsTrans2.append(word)


with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/FastTextenwordsTranslated2es_1.txt', 'wb') as myfile:
      myfile.write("\n".join(eswordsTrans))


with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/FastTextenwordsTranslated2ar_1.txt', 'wb') as myfile:
      myfile.write("\n".join(arwordsTrans))


print sum([True for w in arwordsTrans if len(w)==0]),' out of ',len(arwordsTrans)
'''


if __name__ == '__main__':
    
    if len(sys.argv) < 3:
    	print 'Usage: python pretrain_da.py datasetl datasetr learningRate'
    	print 'Example: python pretrain_da.py gauss basic 0.1'
    	sys.exit()
    
    
    #python autoencoderkeras.py /home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNEleftAttro2enes.txt /home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNErightAttro2enes.txt 0.1
    filepathl= str(sys.argv[1]) 
    filepathr = str(sys.argv[2])
    lr = float(sys.argv[3])
    embeddingsmodel0=loadfasttextmodel('Path To Vectors')
    #embeddingsmodel=loadtransfasttextmodel('Path To Vectors')

embeddingsmodel0=loadtransfasttextmodel('Path To Vectors')
vecten=[]
lng = 'en'
for word in embeddingsmodel0[lng].id2word:
      vecten.append(embeddingsmodel0[lng][word])

#.reshape(-1,300)[0]
vectes=[]
lng = 'es'
for word in embeddingsmodel0[lng].id2word:
      vectes.append(embeddingsmodel0[lng][word])

lng = 'ar'
vectar=[]
embeddingsmodel0[lng] = FastVector(vector_file=filename+'wiki.'+lng+'.vec')
for word in embeddingsmodel0[lng].id2word:
      vectar.append(embeddingsmodel0[lng][word])

vectar=np.asarray(vectar)
vecten=np.asarray(vecten)
vectes=np.asarray(vectes)
    
    #stanford_ner_path = '/home/ahmad/nltk_data/stanford/stanford-ner.jar'
    #os.environ['CLASSPATH'] = stanford_ner_path
    #stanford_classifier = "/home/ahmad/nltk_data/stanford/es/edu/stanford/nlp/models/ner/spanish.ancora.distsim.s512.crf.ser.gz"
    #stes = StanfordNERTagger(stanford_classifier)
    #stanford_classifier = '/home/ahmad/nltk_data/stanford/english.all.3class.distsim.crf.ser.gz'
    #sten = StanfordNERTagger(stanford_classifier)
    #stanford_classifier = "/home/ahmad/nltk_data/stanford/de/edu/stanford/nlp/models/ner/german.conll.hgc_175m_600.crf.ser.gz"
    #stde = StanfordNERTagger(stanford_classifier)
    
filepathl='/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNEleftAttro2enes.txt'
filepathr='/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNErightAttro2enes.txt'


from bs4 import BeautifulSoup
import urllib2
username=['ahmadmu','ahmadmu1','ahmadmu2','ahmadmu3','ahmadmu4','ahmadmu5','ahmadmu6']
pcnt=[]
ncnt=[]
for idx,word in enumerate(arwordsTrans):
  if word=='':
    continue
  
  url = 'http://api.geonames.org/search?q='+enwords[idx]+'&maxRows=1&username='+username[idx%len(username)]
  response = urllib2.urlopen(url)
  webContent = response.read()
  y=BeautifulSoup()
  web=BeautifulSoup(webContent, features="lxml")
  if int(web.geonames.totalresultscount.text)>0:
    pcnt.append(idx)
  else:
    ncnt.append(idx)
  
  if idx % 1000 ==0:
    print idx


print len(pcnt)+len(ncnt),len(arwordsTrans)


arwordsTrans2=[arwordsTrans[idx] for idx in pcnt]

word=arwordsTrans[100]
data_es=[]
data_en=[]
data_ar=[]
wordslist=[]
for word in arwordsTrans2:
  if len(word)<1:
    continue
  
  worden,wordar=word.split(" : ")
  worden=worden.lower().replace(" ","")
  wordar=wordar.replace(" ","")
  
  if '' !=worden and '' !=wordes:
    try:
      if type(worden)!=type(''):
        worden=worden.encode('utf-8')
      
      
      if type(wordar)!=type(''):
        wordar=wordar.encode('utf-8')
      
      
      if embeddingsmodel0['ar'].__contains__(wordar) and embeddingsmodel0['en'].__contains__(worden.encode('utf-8')):
        wordslist.append(worden+ ' : '+wordar)
        data_ar.append(embeddingsmodel0['ar'][wordar])
        data_en.append(embeddingsmodel0['en'][worden])
    except:
      pass

print len(data_en),len(data_ar),len(data_es)



leftNE=[]
with open(filepathl,'r') as myfile:
  leftNE=myfile.readlines()

rightNE=[]
with open(filepathr,'r') as myfile:
  rightNE=myfile.readlines()

leftNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in leftNE]
rightNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in rightNE]
allNE=[]
for _leftNE, _rightNE in zip(leftNE,rightNE):
  word1 = _leftNE[0]
  word2 = _rightNE[0]
  
  if word1+word2 in allNE:
    continue
  else:
    allNE.append(word1+word2)
  
  if word1!= word2:
    continue
  
  if '' !=word1.strip().lower() and '' !=word2.strip().lower():
    try:
      if type(word1)==type(''):
        word1=word1.strip().lower().decode('utf-8')
      else:
        word1=word1.strip().lower()
      
      if type(word2)==type(''):
        word2=word2.strip().lower().decode('utf-8')
      else:
        word2=word2.strip().lower()
      
      if embeddingsmodel0['es'].__contains__(word2) and embeddingsmodel0['en'].__contains__(word1):
        wordslist.append(word1)
        data_es.append(embeddingsmodel0['es'][word2])
        data_en.append(embeddingsmodel0['en'][word1])
    except:
      pass

    
    eswordslist=[]
    for wordidx in range(embeddingsmodel0['es'].n_words):
      if wordidx%10000==0:
        print wordidx
      
      try:
        word1=embeddingsmodel0['es'].id2word[wordidx]
        eswordslist.append(word1)
      except:
          pass
    
    
    allNE=[]
    for wordidx in range(embeddingsmodel0['es'].n_words):
      if wordidx%10000==0:
        print wordidx
      
      try:
        word1=embeddingsmodel0['es'].id2word[wordidx]
        
        data_en.append(embeddingsmodel0['en'][word1])
        data_es.append(embeddingsmodel0['es'][word1])
        wordslist.append(word1)
      except:
          pass
    
    print len(data_en),len(data_es),len(wordslist)
    
data_es=np.asarray(data_es)
data_en=np.asarray(data_en)
data_ar=np.asarray(data_ar)
print data_es.shape,data_en.shape,data_ar.shape
    
    artree = spatial.cKDTree(vectar)
    
    entree = spatial.cKDTree(vecten)
    #dden, iien = entree.query(data_en, k=3, n_jobs=14)
    #dd, ii = entree.query(data_l[0,], k=5, n_jobs=14)
    #embeddingsmodel0['en'].id2word[ii[2]]
    #word='universidad'
    #dd, ii = estree.query(embeddingsmodel0['es'][word], k=5, n_jobs=14)
    #embeddingsmodel0['es'].id2word[ii[0]]
    
    estree = spatial.cKDTree(vectes)
    
    dden, iien = entree.query(data_en, k=2, n_jobs=14)
    ddes, iies = estree.query(data_es, k=2, n_jobs=14)
    ddar, iiar = entree.query(data_ar, k=2, n_jobs=14)
    
    print "1-NN..."
    data_en=list(data_en)
    for ii in iien:
      for i in ii[1:]:
        data_en.append(vecten[i])
    
    data_es=list(data_es)
    for ii in iies:
      for i in ii[1:]:
        data_es.append(vectes[i])
    
    data_en=np.asarray(data_en)
    data_es=np.asarray(data_es)
    
    print len(allNE),data_en.shape,data_es.shape
    
dim=data_en.shape[1]
model = Sequential()
model.add(Dense(9*dim, activation='linear', activity_regularizer=regularizers.l1(0.00000000001),input_shape=(dim,)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(3*dim/4, activation='linear', activity_regularizer=regularizers.l1(0.00000000001)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(2*dim/3, activation='linear', activity_regularizer=regularizers.l1(0.00000000001), name='encoderlayer'))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(3*dim/4, activation='linear', activity_regularizer=regularizers.l1(0.00000000001)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(9*dim/10, activation='linear', activity_regularizer=regularizers.l1(0.00000000001)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(dim))

model.compile(optimizer='Adam', loss=losses.cosine_proximity)
model.fit(data_es, data_en, batch_size=64, epochs=20)
for word in ['barack',wordslist[-1]]:
  print cosine_similarity(model.predict(embeddingsmodel0['es'][word].reshape(1,-1)),embeddingsmodel0['en'][word].reshape(1,-1)),cosine_similarity(embeddingsmodel0['es'][word].reshape(1,-1),embeddingsmodel0['en'][word].reshape(1,-1))

#print cosine_similarity(model.predict(embeddingsmodel0['ar']['?????'].reshape(1,-1)),embeddingsmodel0['en']['barack'].reshape(1,-1))
#print cosine_similarity(embeddingsmodel0['ar']['?????'].reshape(1,-1),embeddingsmodel0['en']['barack'].reshape(1,-1))

print "Transforming model..."
#transformedmodelaren=dict()
lng='es'
transformedmodelaren[lng]=dict()
for word in embeddingsmodel0[lng].id2word:
  transformedmodelaren[lng][word] = model.predict(embeddingsmodel0[lng][word].reshape(1,-1))[0]

lng='en'
transformedmodelaren[lng]=dict()
for word in embeddingsmodel0[lng].id2word:
  transformedmodelaren[lng][word] = embeddingsmodel0[lng][word]

print "Saving model..."
cPickle.dump(transformedmodel, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/FastTextGoogleTransAdamcosLeakyReLU10epcs3Layers.p', 'wb'))




