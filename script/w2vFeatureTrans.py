import cPickle
from scipy import spatial

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

w2vmodel=loadw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')



from fasttext import FastVector

def loadfasttextmodel(filename):
  filename='/home/ahmad/fastText_multilingual/'
  w2v=dict()
  #['en','es','zh','hr','de','fa','ar','fr']['es','en','de']
  for lng in ['en','es']:
    w2v[lng] = FastVector(vector_file=filename+'wiki.'+lng+'.vec')
    #w2v[lng].apply_transform(filename+'alignment_matrices/'+lng+'.txt')
  
  return w2v


embeddingsmodel=loadfasttextmodel('Path To Vectors')


#w2vmodel=loadFastTextmodel('')
#import sys
#sys.getsizeof(w2vmodel)
Alllangpairs=[]
for _cnt in range(0,10):
  Alllangpairs.extend(cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/Alllangpairs40dScldAttro2_'+str(_cnt)+'.p', 'rb')))

'''
matchingwordsleft=[]
matchingwordsright=[]
locs1=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/locs1Attro2.p', 'rb'))
locs2=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/locs2Attro2.p', 'rb'))
print len(locs1),len(locs2),len(Alllangpairs)
for _locs1,_locs2,_lng in zip(locs1,locs2,Alllangpairs):
  if _lng[0] == _lng[1]:
    continue
  
  for l1 in _locs1:
    if l1 in _locs2:
      matchingwordsleft.append(w2vmodel[_lng[0]][l1])
      matchingwordsright.append(w2vmodel[_lng[1]][l1])

del locs1
del locs2
'''

peoples1=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/peoples1Attro2.p', 'rb'))
peoples2=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/peoples2Attro2.p', 'rb'))


lngs=[]
words=[]
matchingwordsleft=[]
matchingwordsright=[]
for _locs1,_locs2,_lng in zip(peoples1,peoples2,Alllangpairs):
  if _lng[0] == _lng[1] or _lng[0] not in ['en','es'] or _lng[1] not in ['en','es']:
    continue
  
  _locs1=_locs1.split(" ")
  _locs2=_locs2.split(" ")
  for l1 in _locs1:
    if l1 in _locs2:
      try:
        l1=l1.strip().lower()
        if type(l1)==type(''):
          l1=l1.decode('utf-8')
        
        if l1 in words:
          continue
        
        lvect=w2vmodel[_lng[0]][l1]
        rvect=w2vmodel[_lng[1]][l1]
        matchingwordsleft.append(lvect)
        matchingwordsright.append(rvect)
        words.append(l1)
        lngs.append(_lng)
      
      except Exception as e: pass

print len(matchingwordsleft),len(matchingwordsright)

del peoples1
del peoples2

orgs1=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/orgs1Attro2.p', 'rb'))
orgs2=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/orgs2Attro2.p', 'rb'))

print len(orgs1),len(orgs2),len(Alllangpairs)
for _locs1,_locs2,_lng in zip(orgs1,orgs2,Alllangpairs):
  if _lng[0] == _lng[1] or _lng[0] not in ['en','es'] or _lng[1] not in ['en','es']:
    continue
  
  
  _locs1=_locs1.split(" ")
  _locs2=_locs2.split(" ")
  for l1 in _locs1:
    if l1 in _locs2:
      try:
        l1=l1.strip().lower()
        if type(l1)==type(''):
          l1=l1.decode('utf-8')
        
        if l1 in words:
          continue
        
        lvect=w2vmodel[_lng[0]][l1]
        rvect=w2vmodel[_lng[1]][l1]
        matchingwordsleft.append(lvect)
        matchingwordsright.append(rvect)
        words.append(l1)
        lngs.append(_lng)
      
      except Exception as e: pass

print len(matchingwordsleft),len(matchingwordsright)
del orgs1
del orgs2


filepathl='/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordNEleftAttro2enes.txt'
with open(filepathl, 'r') as myfile:
        NE1=myfile.readlines()
for i in range(len(NE1)):
  NE1[i]=[n.replace("u\'","").replace("[","").replace("\'","").replace("]","").strip() for n in NE1[i].strip().split(",")]
filepathl='/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordNErightAttro2enes.txt'
with open(filepathl, 'r') as myfile:
        NE2=myfile.readlines()
for i in range(len(NE2)):
  NE2[i]=[n.replace("u\'","").replace("[","").replace("\'","").replace("]","").strip() for n in NE2[i].strip().split(",")]
print len(NE1),len(NE2),len(Alllangpairs)

cnt=0
for _locs1,_locs2,_lng in zip(NE1,NE2,Alllangpairs):
  if len(_locs1)>1 and len(_locs2)>1 and _lng[0] != _lng[1] and _lng[0] in ['en','es'] and _lng[1] in ['en','es']:
    break
    cnt+=1

lngs=[]
words=[]
matchingwordsleft=[]
matchingwordsright=[]

for _locs1,_locs2,_lng in zip(NE1,NE2,Alllangpairs):
  if _lng[0] == _lng[1] or _lng[0] not in ['en','es'] or _lng[1] not in ['en','es']:
    continue
  
  
  #_locs1=_locs1.split(" ")
  #_locs2=_locs2.split(" ")
  for l1 in _locs1:
    if l1 in _locs2:
      try:
        l1=l1.strip().lower()
        if type(l1)==type(''):
          l1=l1.decode('utf-8')
        
        if l1 in words:
          continue
        
        lvect=w2vmodel[_lng[0]][l1]
        rvect=w2vmodel[_lng[1]][l1]
        matchingwordsleft.append(lvect)
        matchingwordsright.append(rvect)
        words.append(l1)
        lngs.append(_lng)
      
      except Exception as e: pass

print len(matchingwordsleft),len(matchingwordsright)

langpr=dict()
for _langpr in lngs:
  if str(_langpr) not in langpr.keys():
    langpr[str(_langpr)]=0
  
  langpr[str(_langpr)]+=1

print langpr

with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/matchingwordsleftAttro2enes.txt','w') as myfile:
  myfile.write("\n".join(map(str,matchingwordsleft)))

with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/matchingwordsrightAttro2enes.txt','w') as myfile:
  myfile.write("\n".join(map(str,matchingwordsright)))
#
#
w2vmodel=loadw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')

##
#Autoencoder part
##

Weights=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/WeightsAttro2enes.p', 'rb'))
Weights=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/WeightsAttro2enDAE.p', 'rb'))
Weights=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/WeightsAttro2enes.p', 'rb'))
Weights=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/WeightsAttro2enesNormCos1000epcs1.p', 'rb'))
W=Weights['W']
b=Weights['b']
Weights=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/WeightsAttro2enesNormCos1000epcs2.p', 'rb'))
W2=Weights['W']
b2=Weights['b']
Weights=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/WeightsAttro2enesNormCos1000epcs3.p', 'rb'))
W3=Weights['W']
b3=Weights['b']

cosine_similarity(transformedmodel['en']['document'][0].reshape(1,-1),transformedmodel['es']['documento'][0].reshape(1,-1))
cosine_similarity(embeddingsmodel['en']['document'].reshape(1,-1),embeddingsmodel['es']['documento'].reshape(1,-1))
cosine_similarity(transformedmodel['en']['document'].reshape(1,-1),embeddingsmodel['es']['documento'].reshape(1,-1))

cosine_similarity(transformedmodel['en']['car'][0].reshape(1,-1),transformedmodel['es']['documento'][0].reshape(1,-1))
cosine_similarity(embeddingsmodel['en']['car'].reshape(1,-1),embeddingsmodel['es']['documento'].reshape(1,-1))

cosine_similarity(transformedmodel['en']['barack'][0].reshape(1,-1),transformedmodel['es']['barack'][0].reshape(1,-1))
cosine_similarity(embeddingsmodel['en']['barack'].reshape(1,-1),embeddingsmodel['es']['barack'].reshape(1,-1))
cosine_similarity(embeddingsmodel['en']['car'].reshape(1,-1),embeddingsmodel['es']['documento'].reshape(1,-1))

cosine_similarity(pred,embeddingsmodel['es']['barack'].reshape(1,-1))

embeddingsmodel['es']['obama'].reshape(1,-1)
transformedmodel['es']['obama'][0].reshape(1,-1)

np.allclose(embeddingsmodel['es']['barack'].reshape(1,-1), transformedmodel['es']['barack'][0].reshape(1,-1))
np.allclose(embeddingsmodel['en']['barack'].reshape(1,-1), data_l[0])
(embeddingsmodel['es']['obama'].reshape(1,-1) == transformedmodel['es']['obama'][0].reshape(1,-1)).all()

cosine_similarity(data_l[0].reshape(1,-1),data_l[1].reshape(1,-1))
transformedmodel['en']['car'][0].reshape(1,-1)
lng='en'
word='car'
caren=model.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))
cosine_similarity(docen0.reshape(1,-1),doces0.reshape(1,-1))
cosine_similarity(docen.reshape(1,-1),doces.reshape(1,-1))

cosine_similarity(car0.reshape(1,-1),doces0.reshape(1,-1))
cosine_similarity(caren.reshape(1,-1),doces.reshape(1,-1))


car16=transformedmodel['en']['car']
transformedmodel=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/FastTextAdamcosLeakyReLU10epcs3Layers.p', 'rb'))
run(0.5)

from sklearn.metrics.pairwise import euclidean_distances
print euclidean_distances(embeddingsmodel['en']['barack'].reshape(1,-1),embeddingsmodel['es']['barack'].reshape(1,-1))
print euclidean_distances(transformedmodel['en']['barack'][0].reshape(1,-1),transformedmodel['es']['barack'][0].reshape(1,-1))
print euclidean_distances(pred,embeddingsmodel['es']['barack'].reshape(1,-1))
pred

for lng in w2vmodel.keys():
  if lng not in ['es']:
    continue
  
  for ky in w2vmodel[lng].keys():
    if len(w2vmodel[lng][ky])< W.shape[0]:
      continue
    
    hx = np.dot(w2vmodel[lng][ky], W) +b
    xr = (1./(1+np.exp(-hx)))
    w2vmodel[lng][ky]=xr



#
# Matrix Trans
#
A=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AAttro2enes.p', 'rb'))
A=cPickle.load(open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AlstsqAttro2enes.p', 'rb'))

for lng in w2vmodel.keys():
  if lng not in ['es','en']:
    continue
  
  for ky in w2vmodel[lng].keys():
    if len(w2vmodel[lng][ky])< W.shape[0]:
      continue
    
    w2vmodel[lng][ky] = np.dot(w2vmodel[lng][ky], A)



#transformedmodel=dict()
lng='en'
#transformedmodel[lng]=dict()
for word in embeddingsmodel[lng].id2word:
      #transformedmodel[lng][word] = encoder.predict(embeddingsmodel[lng][word].reshape(1,-1))
      transformedmodel[lng][word] = model.predict(embeddingsmodel[lng][word].reshape(-1,data_r.shape[1]))
      #transformedmodel[lng][word] = model.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))
      #transformedmodel[lng][word] = encoder.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))

lng='es'
#transformedmodel[lng]=dict()
for word in embeddingsmodel[lng].id2word:
      #transformedmodel[lng][word] = (((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin)
      transformedmodel[lng][word] = (embeddingsmodel[lng][word].reshape(-1,data_r.shape[1]))
      #transformedmodel[lng][word] = encoder.predict(embeddingsmodel[lng][word].reshape(1,-1))#encoder.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))
      #transformedmodel[lng][word] = model.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))
      #transformedmodel[lng][word] = encoder.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))
