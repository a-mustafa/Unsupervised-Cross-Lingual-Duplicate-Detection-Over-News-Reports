import numpy as np
from os import listdir
from os.path import isfile, join
import re
import json
from sklearn.metrics.pairwise import cosine_similarity

langcode={"eng":"en","spa":"es","deu":"de","zho":"zh","ita":"it","fra":"fr","rus":"ru","swe":"sv","nld":"nl","tur":"tr","jpn":"ja","por":"pt","ara":"ar","fin":"fi","ron":"ro","kor":"ko","hrv":"hr","tam":"","hun":"hu","slv":"sl","pol":"pl","srp":"sr","cat":"ca","ukr":"uk"}

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
      w2v[lineparts[0]][wordvector[0]]=list(map(float,wordvector[1:]))
  return w2v

#w2v= loadmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCluster.m_1000+iter_10+window_3+min_count_5+size_40.normalized')
w2v= loadmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')

pos=[]

posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
allfiles = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]

for filenm in allfiles:
  try:
    with open(filenm,"r") as myfile:
      jsonfile=json.load(myfile)
    
    #indices = [i,x for i, x in enumerate(list(jsonfile.keys())) if "-" in x]
    keys = [x for x in list(jsonfile.keys()) if "-" in x]
    #artcle=jsonfile[keys[0]]['articles']['results'][0]
    group1=[]
    for artcle in jsonfile[keys[0]]['articles']['results']:
      wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
      w2vvectors=[]
      for word in wordslist:
        if word.strip().lower() in w2v[langcode[artcle['lang']]].keys() and '' !=word.strip().lower():
          w2vvectors.append(w2v[langcode[artcle['lang']]][word.strip().lower()])
      
      group1.append(np.mean(w2vvectors,axis=0))
    
    group2=[]
    for artcle in jsonfile[keys[1]]['articles']['results']:
      wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
      w2vvectors=[]
      for word in wordslist:
        if word.strip().lower() in w2v[langcode[artcle['lang']]].keys() and '' !=word.strip().lower():
          w2vvectors.append(w2v[langcode[artcle['lang']]][word.strip().lower()])
      
      group2.append(np.mean(w2vvectors,axis=0))
    
    cossim=cosine_similarity(group1,group2)
    
    for x1 in range(len(cossim)):
      for x2 in range(x1,len(cossim[x1])):
        pos.append(cossim[x1][x2])
  except: 
    pass

with open('/home/ahmad/duplicate-detection/eventregistrydata/posCCA.txt',"w") as myfile:
  myfile.write(str(pos))

'''
filenm = allfiles[0]
np.mean(np.array(w2vvectors),axis=0)
for word in wordslist:
  w2v[langcode[artcle['lang']]][re.sub(r'\W+', '', word).lower()]

with open('/home/ahmad/duplicate-detection/eventregistrydata/positive/3.json',"r") as myfile:
  jsonfile=json.load(myfile)

with open('/home/ahmad/duplicate-detection/eventregistrydata/negative/1.json',"r") as myfile:
  jsonfile=json.load(myfile)

cosine_similarity(dupgroup)

from scipy.spatial.distance import cosine 
for x1 in range(len(dupgroup)):
  for x2 in range(x1,len(dupgroup)):
    print(cosine(dupgroup[x1],dupgroup[x2]))

'''
neg=[]
allfiles = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]

for filenm in allfiles:
  try:
    with open(filenm,"r") as myfile:
      jsonfile=json.load(myfile)
    
    keys = [x for x in list(jsonfile.keys()) if "-" in x]
    group1=[]
    for artcle in jsonfile[keys[0]]['articles']['results']:
      wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
      w2vvectors=[]
      for word in wordslist:
        if word.strip().lower() in w2v[langcode[artcle['lang']]].keys() and '' !=word.strip().lower():
          w2vvectors.append(w2v[langcode[artcle['lang']]][word.strip().lower()])
      
      group1.append(np.mean(w2vvectors,axis=0))
    
    group2=[]
    for artcle in jsonfile[keys[1]]['articles']['results']:
      wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
      w2vvectors=[]
      for word in wordslist:
        if word.strip().lower() in w2v[langcode[artcle['lang']]].keys() and '' !=word.strip().lower():
          w2vvectors.append(w2v[langcode[artcle['lang']]][word.strip().lower()])
      
      group2.append(np.mean(w2vvectors,axis=0))
    
    cossim2=cosine_similarity(group1,group2)
    
    for x1 in range(len(cossim2)):
      for x2 in range(x1,len(cossim2[x1])):
        neg.append(cossim2[x1][x2])
  except: 
    pass

with open('/home/ahmad/duplicate-detection/eventregistrydata/negCCA.txt',"w") as myfile:
  myfile.write(str(neg))