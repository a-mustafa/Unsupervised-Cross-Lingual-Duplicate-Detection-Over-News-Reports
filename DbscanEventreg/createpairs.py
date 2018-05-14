'''Train a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''
'''
from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
'''

import numpy as np
import sys
import random
import time
from os import listdir
from os.path import isfile, join
import re
import json
from sklearn.metrics.pairwise import cosine_similarity

def loadmodel(filename):
  w2v=dict()
  #filename='fifty_nine.table5.multiCluster.m_1000+iter_10+window_3+min_count_5+size_40.normalized'
  #filename='/home/ahmad/duplicate-detection/multilingual-embedding/three.table4.multiSkip.iter_10+window_3+min_count_5+size_40.normalized'
  a=0
  with open(filename, "r") as myfile:
    for line in myfile:
      lineparts=line.strip().split(":")
      wordvector=lineparts[1].split(" ")
      if lineparts[0] not in w2v.keys():
          w2v[lineparts[0]]=dict()
      
      #w2v[lineparts[0]][wordvector[0]]=map(float,wordvector[1:])
      w2v[lineparts[0]][wordvector[0].decode('utf-8')]=list(map(float,wordvector[1:]))
  return w2v

def loadunifiedw2vmodel(filename):
  w2v=dict()
  #filename='fifty_nine.table5.multiCluster.m_1000+iter_10+window_3+min_count_5+size_40.normalized'
  #filename='/home/ahmad/duplicate-detection/multilingual-embedding/three.table4.multiSkip.iter_10+window_3+min_count_5+size_40.normalized'
  a=0
  with open(filename, "r") as myfile:
    for line in myfile:
      lineparts=line.strip().split(":")
      wordvector=lineparts[1].split(" ")
      '''
      if lineparts[0] not in w2v.keys():
          w2v[lineparts[0]]=dict()
      '''
      #w2v[lineparts[0]][wordvector[0]]=map(float,wordvector[1:])
      w2v[wordvector[0].decode('utf-8')]=list(map(float,wordvector[1:]))
  return w2v

def create_CMF_pairs(w2vmodel,breaks,nbreaks=[5,10,15]):
  langcode={"eng":"en","spa":"es","deu":"de","zho":"zh","ita":"it","fra":"fr","rus":"ru","swe":"sv","nld":"nl","tur":"tr","jpn":"ja","por":"pt","ara":"ar","fin":"fi","ron":"ro","kor":"ko","hrv":"hr","tam":"","hun":"hu","slv":"sl","pol":"pl","srp":"sr","cat":"ca","ukr":"uk"}
  #breaks=CMF_init(w2vmodel, nbreaks)
  
  pairs=[]
  labels = []
  posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
  negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
  allposfiles = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
  allnegfiles = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]
  pospairs=[]
  print("creating positive pairs:")
  for idx,Pfilenm in enumerate(allposfiles):
    try:
      with open(Pfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      group1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix1=[]
        for word in wordslist:
          if '' !=word.strip().lower():
            try:
              w2vmatrix1.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
            except:
              pass
        
        group1.append(np.array(w2vmatrix1))
      
      group2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix2=[]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              w2vmatrix2.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
            except:
              pass
        
        group2.append(np.array(w2vmatrix2))
      
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          #pospairs.append(group1[x1],group2[x2])
          pospairs.append([CMF(group1[x1],breaks),CMF(group2[x2],breaks)])
      
      sys.stdout.write("\r")
      sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allposfiles)), 100*idx/len(allposfiles)))
      sys.stdout.flush()
      
    except:
      pass
  
  
  print('\ncreating negative pairs...')
  
  negpairs=[]
  for idx,Nfilenm in enumerate(allnegfiles):
    try:
      with open(Nfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      
      group1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix1=[]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              w2vmatrix1.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
            except:
              pass
        
        group1.append(np.array(w2vmatrix1))
      
      group2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix2=[]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and
            try:
              w2vmatrix2.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
            except:
              pass
        
        group2.append(np.array(w2vmatrix2))
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          negpairs.append([CMF(group1[x1],breaks),CMF(group2[x2],breaks)])#(group1[x1],group2[x2])
      
      sys.stdout.write("\r")
      sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allnegfiles)), 100*idx/len(allnegfiles)))
      sys.stdout.flush()
    except:
      pass
  
  print("\nShuffling...")
  for pospair,negpair in zip(pospairs,negpairs):
    pairs.append(pospair)
    pairs.append(negpair)
    labels.append(1)
    labels.append(0)
  
  return np.array(pairs),np.array(labels)

def CMF_init(w2vmodel,nbreaks=[5,10,15]):
  print("Computing Word2vec statistics..")
  w2vmodelmatrix=[]
  for lang in w2vmodel.keys():
    for wrd in w2vmodel[lang]:
      if len(w2vmodel[lang][wrd])>0:
        w2vmodelmatrix.append(w2vmodel[lang][wrd])
  
  w2vmodelmatrix=np.asarray(w2vmodelmatrix)
  breaks=[]
  for topic in range(0,w2vmodelmatrix.shape[1]):
    topicbreaks=[]
    for binsize in nbreaks:
      topicbreaks.append(np.linspace(min(w2vmodelmatrix[:,topic]),max(w2vmodelmatrix[:,topic]),binsize))
    breaks.append(topicbreaks)
    sys.stdout.write("\r")
    sys.stdout.write("[%-100s] %d%%" % ('='*(100*topic/w2vmodelmatrix.shape[1]), 100*topic/w2vmodelmatrix.shape[1]))
    sys.stdout.flush()
  
  print("\n")
  return breaks

#CMF(group1[x1],breaks)
#data=group1[x1]
def CMF(data, breaks=4):
  newdata=len(breaks)*[[]]
  for d in range(data.shape[1]):
    for brk in breaks[d]:
      values, base = np.histogram(data[:,d], bins=list(brk), normed=True)
      cumulative = np.cumsum(values)
      newdata[d].extend(list(cumulative))
    return np.array(newdata)
#
#
#

def create_words_pairs(w2vmodel,fromfileidx,tofileidx):
  langcode={"eng":"en","spa":"es","deu":"de","zho":"zh","ita":"it","fra":"fr","rus":"ru","swe":"sv","nld":"nl","tur":"tr","jpn":"ja","por":"pt","ara":"ar","fin":"fi","ron":"ro","kor":"ko","hrv":"hr","tam":"","hun":"hu","slv":"sl","pol":"pl","srp":"sr","cat":"ca","ukr":"uk"}
  #w2vmodel= loadmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
  pairs=[]
  labels = []
  posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
  negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
  allposfiles = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
  allnegfiles = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]
  pospairs=[]
  print("creating positive pairs:")
  for idx,Pfilenm in enumerate(allposfiles[fromfileidx:tofileidx]):
    try:
      with open(Pfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      group1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix1=[langcode[artcle['lang']]]
        for word in wordslist:
          if '' !=word.strip().lower():
            try:
              w2vmatrix1.append(word.strip().lower())
            except:
              pass
        
        group1.append(w2vmatrix1)
      
      group2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix2=[langcode[artcle['lang']]]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              w2vmatrix2.append(word.strip().lower())
            except:
              pass
        
        group2.append(w2vmatrix2)
      
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          pospairs.append([group1[x1],group2[x2]])
          #pospairs.append([CMF(group1[x1],breaks),CMF(group2[x2],breaks)])
      
      
      
      sys.stdout.write("\r")
      sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allposfiles[fromfileidx:tofileidx])), 100*idx/len(allposfiles[fromfileidx:tofileidx])))
      sys.stdout.flush()
      
    except:
      pass
  
  
  print('\ncreating negative pairs...')
  
  negpairs=[]
  for idx,Nfilenm in enumerate(allnegfiles[fromfileidx:tofileidx]):
    try:
      with open(Nfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      
      group1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix1=[langcode[artcle['lang']]]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              w2vmatrix1.append(word.strip().lower())
            except:
              pass
        
        group1.append(w2vmatrix1)
      
      group2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix2=[langcode[artcle['lang']]]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and
            try:
              w2vmatrix2.append(word.strip().lower())
            except:
              pass
        
        group2.append(w2vmatrix2)
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          negpairs.append([group1[x1],group2[x2]])
          #negpairs.append([CMF(group1[x1],breaks),CMF(group2[x2],breaks)])#(group1[x1],group2[x2])
      
      
      sys.stdout.write("\r")
      sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allnegfiles[fromfileidx:tofileidx])), 100*idx/len(allnegfiles[fromfileidx:tofileidx])))
      sys.stdout.flush()
    except:
      pass
  
  print("\nShuffling...")
  for pospair,negpair in zip(pospairs,negpairs):
    pairs.append(pospair)
    pairs.append(negpair)
    labels.append(1)
    labels.append(0)
  
  return pairs,labels

#
#
#

def create_w2v_pairs(w2vmodel,fromfileidx,tofileidx):
  langcode={"eng":"en","spa":"es","deu":"de","zho":"zh","ita":"it","fra":"fr","rus":"ru","swe":"sv","nld":"nl","tur":"tr","jpn":"ja","por":"pt","ara":"ar","fin":"fi","ron":"ro","kor":"ko","hrv":"hr","tam":"","hun":"hu","slv":"sl","pol":"pl","srp":"sr","cat":"ca","ukr":"uk"}
  #w2vmodel= loadmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
  pairs=[]
  wpairs=[]
  langpairs=[]
  labels = []
  posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
  negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
  allposfiles = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
  allnegfiles = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]
  pospairs=[]
  poswordpairs=[]
  poslangpairs=[]
  print("creating positive pairs:")
  for idx,Pfilenm in enumerate(allposfiles[fromfileidx:tofileidx]):
    try:
      with open(Pfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      lgroup1=[]
      group1=[]
      wgroup1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix1=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower():
            try:
              w2vmatrix1.append(w2vmodel[word.strip().lower()])
              #w2vmatrix1.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
              wlist.append(word.strip().lower())
            except:
              pass
        
        lgroup1.append(langcode[artcle['lang']])
        group1.append(np.array(w2vmatrix1))
        wgroup1.append(wlist)
      
      lgroup2=[]
      group2=[]
      wgroup2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix2=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              w2vmatrix2.append(w2vmodel[word.strip().lower()])
              #w2vmatrix2.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
              wlist.append(word.strip().lower())
            except:
              pass
        
        lgroup2.append(langcode[artcle['lang']])
        group2.append(np.array(w2vmatrix2))
        wgroup2.append(wlist)
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          pospairs.append([np.array(group1[x1]),np.array(group2[x2])])
          poswordpairs.append([wgroup1[x1],wgroup2[x2]])
          poslangpairs.append([lgroup1[x1],lgroup2[x2]])
      
      
      
      sys.stdout.write("\r")
      sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allposfiles[fromfileidx:tofileidx])), 100*idx/len(allposfiles[fromfileidx:tofileidx])))
      sys.stdout.flush()
      
    except:
      pass
  
  
  print('\ncreating negative pairs...')
  
  neglangpairs=[]
  negpairs=[]
  negwordpairs=[]
  for idx,Nfilenm in enumerate(allnegfiles[fromfileidx:tofileidx]):
    try:
      with open(Nfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      
      lgroup1=[]
      group1=[]
      wgroup1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix1=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              w2vmatrix1.append(w2vmodel[word.strip().lower()])
              #w2vmatrix1.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
              wlist.append(word.strip().lower())
            except:
              pass
        
        lgroup1.append(langcode[artcle['lang']])
        group1.append(np.array(w2vmatrix1))
        wgroup1.append(wlist)
      
      lgroup2=[]
      group2=[]
      wgroup2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix2=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and
            try:
              w2vmatrix2.append(w2vmodel[word.strip().lower()])
              #w2vmatrix2.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
              wlist.append(word.strip().lower())
            except:
              pass
        
        lgroup2.append(langcode[artcle['lang']])
        group2.append(np.array(w2vmatrix2))
        wgroup2.append(wlist)
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          negpairs.append([np.array(group1[x1]),np.array(group2[x2])])
          negwordpairs.append([wgroup1[x1],wgroup2[x2]])
          neglangpairs.append([lgroup1[x1],lgroup2[x2]])
      
      
      sys.stdout.write("\r")
      sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allnegfiles[fromfileidx:tofileidx])), 100*idx/len(allnegfiles[fromfileidx:tofileidx])))
      sys.stdout.flush()
    except:
      pass
  
  print("\nShuffling...")
  print len(pospairs),len(negpairs),len(poswordpairs),len(negwordpairs),len(poslangpairs),len(neglangpairs)
  for pospair,negpair,poswpair,negwpair,poslpair,neglpair in zip(pospairs,negpairs,poswordpairs,negwordpairs,poslangpairs,neglangpairs):
    pairs.append(pospair)
    pairs.append(negpair)
    labels.append(1)
    labels.append(0)
    wpairs.append(poswpair)
    wpairs.append(negwpair)
    langpairs.append(poslpair)
    langpairs.append(neglpair)
  
  return np.array(pairs),np.array(labels), wpairs,langpairs

#
w2vpairs,labels,wordspairs,langpairs=create_w2v_pairs(w2vmodel,frm,frm+10)
print len(w2vpairs),len(labels),len(wordspairs),len(langpairs)
w2vpairs1,labels1,wordspairs1,langpairs1=create_w2v_pairs(unifiedw2vmodel,frm,frm+10)
print len(w2vpairs1),len(labels1),len(wordspairs1),len(langpairs1)
#
#

def create_lang_pairs(w2vmodel,fromfileidx,tofileidx):
  langcode={"eng":"en","spa":"es","deu":"de","zho":"zh","ita":"it","fra":"fr","rus":"ru","swe":"sv","nld":"nl","tur":"tr","jpn":"ja","por":"pt","ara":"ar","fin":"fi","ron":"ro","kor":"ko","hrv":"hr","tam":"","hun":"hu","slv":"sl","pol":"pl","srp":"sr","cat":"ca","ukr":"uk"}
  #w2vmodel= loadmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
  pairs=[]
  labels = []
  posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
  negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
  allposfiles = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
  allnegfiles = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]
  pospairs=[]
  print("creating positive pairs:")
  for idx,Pfilenm in enumerate(allposfiles):
    try:
      with open(Pfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      group1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        '''
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix1=[]
        for word in wordslist:
          if '' !=word.strip().lower():
            try:
              w2vmatrix1.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
            except:
              pass
        '''
        
        group1.append(langcode[artcle['lang']])
      
      group2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        '''
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix2=[]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              w2vmatrix2.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
            except:
              pass
        '''
        group2.append(langcode[artcle['lang']])
      
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          pospairs.append([group1[x1],group2[x2]])
          #pospairs.append([CMF(group1[x1],breaks),CMF(group2[x2],breaks)])
      
      
      
      sys.stdout.write("\r")
      sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allposfiles)), 100*idx/len(allposfiles)))
      sys.stdout.flush()
      
    except:
      pass
  
  
  print('\ncreating negative pairs...')
  
  negpairs=[]
  for idx,Nfilenm in enumerate(allnegfiles):
    try:
      with open(Nfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      
      group1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        '''
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix1=[]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              w2vmatrix1.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
            except:
              pass
        '''
        group1.append(langcode[artcle['lang']])
      
      group2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        '''
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix2=[]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and
            try:
              w2vmatrix2.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
            except:
              pass
        '''
        group2.append(langcode[artcle['lang']])
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          negpairs.append([group1[x1],group2[x2]])
          #negpairs.append([CMF(group1[x1],breaks),CMF(group2[x2],breaks)])#(group1[x1],group2[x2])
      
      
      sys.stdout.write("\r")
      sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allnegfiles)), 100*idx/len(allnegfiles)))
      sys.stdout.flush()
    except:
      pass
  
  print("\nShuffling...")
  for pospair,negpair in zip(pospairs,negpairs):
    pairs.append(pospair)
    pairs.append(negpair)
    labels.append(1)
    labels.append(0)
  
  return pairs,labels

#
#
#

def create_norm_pairs(w2vmodel):
  langcode={"eng":"en","spa":"es","deu":"de","zho":"zh","ita":"it","fra":"fr","rus":"ru","swe":"sv","nld":"nl","tur":"tr","jpn":"ja","por":"pt","ara":"ar","fin":"fi","ron":"ro","kor":"ko","hrv":"hr","tam":"","hun":"hu","slv":"sl","pol":"pl","srp":"sr","cat":"ca","ukr":"uk"}
  
  pairs=[]
  labels = []
  posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
  negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
  allposfiles = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
  allnegfiles = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]
  pospairs=[]
  print("creating positive pairs:")
  for idx,Pfilenm in enumerate(allposfiles):
    try:
      with open(Pfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      group1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix1=[]
        for word in wordslist:
          if '' !=word.strip().lower():
            try:
              w2vmatrix1.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
            except:
              pass
        
        group1.append(np.linalg.norm(np.array(w2vmatrix1)))
       
      group2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix2=[]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              w2vmatrix2.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
            except:
              pass
        
        group2.append(np.linalg.norm(np.array(w2vmatrix2)))
      
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          pospairs.append([group1[x1],group2[x2]])#([np.array(group1[x1]),np.array(group2[x2])])
          #pospairs.append([CMF(group1[x1],breaks),CMF(group2[x2],breaks)])
      
      
      
      sys.stdout.write("\r")
      sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allposfiles)), 100*idx/len(allposfiles)))
      sys.stdout.flush()
      
    except:
      pass
  
  
  print('\ncreating negative pairs...')
  
  negpairs=[]
  for idx,Nfilenm in enumerate(allnegfiles):
    try:
      with open(Nfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      
      group1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix1=[]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              w2vmatrix1.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
            except:
              pass
        
        group1.append(np.linalg.norm(np.array(w2vmatrix1)))
      
      group2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix2=[]
        for word in wordslist:
          if '' !=word.strip().lower(): #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and
            try:
              w2vmatrix2.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
            except:
              pass
        
        group2.append(np.linalg.norm(np.array(w2vmatrix2)))
      
      for x1 in range(len(group1)):
        for x2 in range(len(group2)):
          negpairs.append([group1[x1],group2[x2]])
          #negpairs.append([CMF(group1[x1],breaks),CMF(group2[x2],breaks)])#(group1[x1],group2[x2])
      
      
      sys.stdout.write("\r")
      sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allnegfiles)), 100*idx/len(allnegfiles)))
      sys.stdout.flush()
    except:
      pass
  
  print("\nShuffling...")
  for pospair,negpair in zip(pospairs,negpairs):
    pairs.append(pospair)
    pairs.append(negpair)
    labels.append(1)
    labels.append(0)
  
  return np.array(pairs),np.array(labels)


#docpairs,duplabels=create_w2v_pairs(None)
#docpairs,duplabels=create_norm_pairs(w2vmodel)

#a=create_norm_pairs(None)
w2vmodel= loadmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
unifiedw2vmodel=loadunifiedw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
unifiedw2vmodel=loadunifiedw2vmodel(w2vmodelpath)

docpairs,duplabels=create_w2v_pairs(w2vmodel,frm,to)
alllangpairs,alllbl=create_lang_pairs(w2vmodel,frm,to)
docpairs,duplabels=create_norm_pairs(w2vmodel)

breaks=CMF_init(w2vmodel, [5,10,15])
pairs,labels=create_CMF_pairs(w2vmodel,breaks, nbreaks=[5,10,15])
#pairs,labels=create_w2v_pairs(w2vmodel)
import cPickle
#fnm='/home/ahmad/duplicate-detection/eventregistrydata/cmfpairs-b.5.10.15.dat'
fnm='/home/ahmad/duplicate-detection/eventregistrydata/w2vpairs-b.5.10.15.dat'
mat=[pairs,labels]
cPickle.dump( mat, open( fnm, "wb" ) )

#np.save(fnm, mat)

with open('svm_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)

with open('svm_classifier.pkl', 'rb') as fid:
    gnb_loaded = cPickle.load(fid)