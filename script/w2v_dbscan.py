from nltk.corpus import stopwords
import gensim
import os
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
from datetime import datetime,date,timedelta
#import load
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import manifold

def get_doc_list(sentencesfile):
  dataText=[]
  f = open(sentencesfile, 'r')
  for d in f:
    dataText.append(d.decode('utf-8'))
  
  return dataText


def loadmodel(filename):
  w2v=dict()
  #filename='fifty_nine.table5.multiCluster.m_1000+iter_10+window_3+min_count_5+size_40.normalized'
  #filename='/home/ahmad/duplicate-detection/multilingual-embedding/three.table4.multiSkip.iter_10+window_3+min_count_5+size_40.normalized'
  #filename='/home/ahmad/duplicate-detection/multilingual-embedding/twelve.table4.translation_invariance.window_3+size_40.normalized'
  with open(filename, "r") as myfile:
    for line in myfile:
      lineparts=line.strip().split(":")
      if len(lineparts)<=1:
        continue
      wordvector=lineparts[1].split(" ")
      if lineparts[0] not in w2v.keys():
          w2v[lineparts[0]]=dict()
      
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



def clustering_purity(_w2vpairs,dbscan_eps=0.5, dbscan_minPts=5):
  numclusters=[]
  clustersdist=[]
  pureclustersratio=[]
  _w2vpairs=w2vpairs
  for pridx in range(len(_w2vpairs)):
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
    
    
    clustersdist.append(clusters)#pairclusters)
    numclusters.append([n_pure_cl,n_mixed_cl,n_noise_cl])
    pureclustersratio.append(1.0*n_pure_cl/(n_pure_cl+n_mixed_cl+0.00001))
  
  return numclusters, pureclustersratio, clustersdist


def clustering_purity(w2vpairs,dbscan_eps=0.5, dbscan_minPts=5):
  numclusters=[]
  clustersdist=[]
  pureclustersratio=[]
  
  for pridx in range(len(w2vpairs)):
    if w2vpairs[pridx][0].size ==0 or w2vpairs[pridx][1].size ==0:
      numclusters.append([-100000,-100000,-100000])
      pureclustersratio.append(-100000)
      clustersdist.append([])
      continue
    
    X=np.vstack((w2vpairs[pridx][0],w2vpairs[pridx][1]))
    Y=[1]*w2vpairs[pridx][0].shape[0]+[2]*w2vpairs[pridx][1].shape[0]
    
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts).fit(X)
    ll=np.unique(db.labels_)
    n_pure_cl=0
    n_noise_cl=0
    n_mixed_cl=0
    pairclusters=[]
    for _ll in ll:
      if _ll == -1:
        n_noise_cl+=1
        continue
      
      idx=np.where(db.labels_==_ll)[0]
      lblsofcl=[Y[_idx] for _idx in idx]
      pairclusters.append(lblsofcl)
      if len(set(lblsofcl))>1:
        n_mixed_cl+=1
      else:
        n_pure_cl+=1
    
    clustersdist.append(pairclusters)
    numclusters.append([n_pure_cl,n_mixed_cl,n_noise_cl])
    pureclustersratio.append(1.0*n_pure_cl/(n_pure_cl+n_mixed_cl+0.00001))
  
  return numclusters, pureclustersratio, clustersdist

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item).lower())
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    #stems = stem_tokens(tokens, stemmer)
    return tokens

def distancetfidfpairs(tfidfpairs):
  cosdistance=[]
  for pridx in range(len(tfidfpairs)): 
    cosdistance.append(spatial.distance.cosine(tfidfpairs[pridx][0],tfidfpairs[pridx][1]))
    #cosine_similarity(pairs[pridx][0],pairs[pridx][1])
  
  return cosdistance

def jaccarddistancepairs(binarypairs):
  jaccarddistance=[]
  for pridx in range(len(binarypairs)):
    val=spatial.distance.jaccard(binarypairs[pridx][0],binarypairs[pridx][1])
    if np.isnan(val):
      jaccarddistance.append(0)
    else:
      jaccarddistance.append(val)
  
  return jaccarddistance



'''
import cPickle
cPickle.dump(labels[:10], open('/home/ahmad/duplicate-detection/siameseCMF/labels.p', 'wb'))

import cPickle
reader = cPickle.load(open('/home/ahmad/duplicate-detection/siameseCMF/labels.p', 'rb'))
'''

if __name__ == '__main__':

sentences='/home/ahmad/duplicate-detection/atrocitiesdata/LocationsList4.txt'
LocationsList=get_doc_list(sentences)

sentences='/home/ahmad/duplicate-detection/atrocitiesdata/PersonsList3.txt'
PersonsList=get_doc_list(sentences)

sentences='/home/ahmad/duplicate-detection/atrocitiesdata/OrgsList3.txt'
OrgsList=get_doc_list(sentences)

sentences='/home/ahmad/duplicate-detection/atrocitiesdata/DatesList3.txt'
DatesList=get_doc_list(sentences)

sentencesfile='/home/ahmad/duplicate-detection/atrocitiesdata/TextList3.txt'
dataText=get_doc_list(sentencesfile)

PersonsList.append(" ")
OrgsList.append(" ")
print len(dataText), len(DatesList), len(LocationsList), len(ClusterLabel), len(PersonsList), len(OrgsList), len(KeysList)

dataText_vectorizer = TfidfVectorizer(analyzer = "word",tokenizer = tokenize, preprocessor = None, stop_words = 'english', max_features = 1000) #, ngram_range=(1, 1)
dataText_features = dataText_vectorizer.fit_transform(dataText)
dataText_features = dataText_features.toarray()

Location_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, binary=True, preprocessor = None, stop_words = None, max_features = 1000) #, ngram_range=(1, 1)
Location_features = Location_vectorizer.fit_transform(LocationsList)
Location_features = Location_features.toarray()

Person_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, binary=True, preprocessor = None, stop_words = None, max_features = 1000) #, ngram_range=(1, 1)
Person_features = Person_vectorizer.fit_transform(PersonsList)
Person_features = Person_features.toarray()

Orgs_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, preprocessor = None, binary=True, stop_words = None, max_features = 1000) #, ngram_range=(1, 1)
Orgs_features = Orgs_vectorizer.fit_transform(OrgsList)
Orgs_features = Orgs_features.toarray()

Dates_vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, preprocessor = None, binary=True, stop_words = None, max_features = 1000) #, ngram_range=(1, 1)
Dates_features = Dates_vectorizer.fit_transform(DatesList)
Dates_features = Dates_features.toarray()

NEsfeatures= np.hstack((Location_features,Person_features,Orgs_features,Dates_features))

dataText_vectorizer = TfidfVectorizer(analyzer = "word",tokenizer = tokenize, preprocessor = None, max_features = 1000) #, ngram_range=(1, 1)
dataText_features = dataText_vectorizer.fit_transform(dataText)
dataText_features = dataText_features.toarray()
w2vmodelpath='/home/ahmad/duplicate-detection/multilingual-embedding/twelve.table4.translation_invariance.window_3+size_40.normalized'
#w2vmodelpath='/home/ahmad/duplicate-detection/multilingual-embedding/twelve.table4.translation_invariance.size_512+window_3.normalized'
#w2vmodelpath='/home/ahmad/duplicate-detection/multilingual-embedding/twelve.table4.multiSkip.size_512+w_5+it_10.normalized'
w2vmodel=loadmodel(w2vmodelpath)
en_stop = set(stopwords.words('english'))
w2vdata=[]
wordsdata=[]
for artcle in dataText:
  wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle)
  #wordslist=set(wordslist)
  w2vmatrix=[]
  wlist=[]
  for word in wordslist:
    if '' !=word.strip().lower() and word.strip().lower() not in en_stop:
      try:
        #w2vmatrix.append(w2vmodel[word.strip().lower()])
        w2vmatrix.append(w2vmodel['en'][word.strip().lower()])
        wlist.append(word.strip().lower())
      except:
        pass
  
  w2vdata.append(np.array(w2vmatrix))
  wordsdata.append(wlist)

sentences='/home/ahmad/duplicate-detection/atrocitiesdata/ClusterLabel3.txt'
ClusterLabel=[]
f = open(sentences, 'r')
for d in f:
  ClusterLabel.append(int(d.strip()))

sentences='../atrocitiesdata/KeysList4.txt'
KeysList=[]
f = open(sentences, 'r')
for d in f:
  KeysList.append(d.decode('utf-8').strip())

print "len(ClusterLabel), len(KeysList), len(w2vdata), len(wordsdata)"
print len(ClusterLabel), len(KeysList), len(w2vdata), len(wordsdata)

rootdir='/home/ahmad/duplicate-detection/atrocitiesdata/OrderedDocuments/'
dirsList=next(os.walk(rootdir))[1]
for d in range(len(dirsList)):
  dirsList[d]= datetime.strptime(dirsList[d], '%d_%b_%Y')

dirsList.sort()
dirsList=[os.path.join(rootdir, date.strftime(s,'%d_%b_%Y')) for s in dirsList]
n=dataText_features.shape[0]
#Allnumclusters=[]
#Allpureclustersratio=[]
#Allclustersdist=[]
Allcosdist=[]
AllNEdist=[]
Allisdup_labels=[]
Trainingindexes=[]
for day in range(len(dirsList)):
  fs=next(os.walk(dirsList[day]))[2]
  for i, j in enumerate(fs):
    key=j.split('.')[0]
    if key in KeysList:
      instanceindex=KeysList.index(key)
      Trainingindexes.append(instanceindex)
  
  if day>5 and day % 5 == 0:
    documents = [w2vdata[i] for i in Trainingindexes]
    docterms = [wordsdata[i] for i in Trainingindexes]
    clusterslabels=[ClusterLabel[i] for i in Trainingindexes]
    Trmask=np.zeros(n,dtype='bool')
    Trmask[Trainingindexes]=True
    datax=dataText_features[Trmask,:]
    NEs=NEsfeatures[Trmask,:]
    w2vpairs=[]
    isdup_labels=[]
    wordpairs=[]
    Tfidfpairs=[]
    NEspairs=[]
    for x1 in range(len(documents)):
      for x2 in range(x1,len(documents)):
        w2vpairs.append([documents[x1],documents[x2]])
        wordpairs.append([docterms[x1],docterms[x2]])
        Tfidfpairs.append([datax[x1],datax[x2]])
        NEspairs.append([NEs[x1],NEs[x2]])
        if clusterslabels[x1]==clusterslabels[x2]:
          isdup_labels.append(1)
        else:
          isdup_labels.append(0)
    
    #numclusters,pureclustersratio,clustersdist=clustering_purity(w2vpairs,dbscan_eps=0.5, dbscan_minPts=5)
    #Allclustersdist.extend(clustersdist)
    #Allnumclusters.extend(numclusters)
    #Allpureclustersratio.extend(pureclustersratio)
    Allcosdist.extend(distancetfidfpairs(Tfidfpairs))
    AllNEdist.extend(jaccarddistancepairs(NEspairs))
    Allisdup_labels.extend(isdup_labels)
    Trainingindexes=[]


import cPickle
cPickle.dump(Allclustersdist, open('/home/ahmad/duplicate-detection/atrocitiesdata/AllclustersdistAttro.p', 'wb'))
cPickle.dump(Allnumclusters, open('/home/ahmad/duplicate-detection/atrocitiesdata/AllnumclustersAttro.p', 'wb'))
cPickle.dump(Allpureclustersratio, open('/home/ahmad/duplicate-detection/atrocitiesdata/AllpureclustersratioAttro.p', 'wb'))
cPickle.dump(Allcosdist, open('/home/ahmad/duplicate-detection/atrocitiesdata/AllcosdistAttro.p', 'wb'))
cPickle.dump(AllNEdist, open('/home/ahmad/duplicate-detection/atrocitiesdata/AllNEdistAttro.p', 'wb'))
cPickle.dump(Allisdup_labels, open('/home/ahmad/duplicate-detection/atrocitiesdata/Allisdup_labelsAttro.p', 'wb'))
#obj = cPickle.load(open('save.p', 'rb'))
Allnumclusters=cPickle.load(open('/home/ahmad/duplicate-detection/atrocitiesdata/Attro.p', 'rb'))
Allpureclustersratio=cPickle.load(open('/home/ahmad/duplicate-detection/atrocitiesdata/Attro.p', 'rb'))
Allclustersdist=cPickle.load(open('/home/ahmad/duplicate-detection/atrocitiesdata/Attro.p', 'rb'))
Allcosdist=cPickle.load(open('/home/ahmad/duplicate-detection/atrocitiesdata/Attro.p', 'rb'))
AllNEdist=cPickle.load(open('/home/ahmad/duplicate-detection/atrocitiesdata/Attro.p', 'rb'))
Allisdup_labels=cPickle.load(open('/home/ahmad/duplicate-detection/atrocitiesdata/Attro.p', 'rb'))
#
#
#### THE END
#
#

#
#### Calc entropy
#
import math
Entropy=[]
for clutering in Allclustersdist:
  H_Omega=0
  flat_clstrs_len = len([item for sublist in clutering for item in sublist])
  for clstr in clutering:
    ndoc1= sum([True for cl in clstr if cl == 1])
    ndoc2= sum([True for cl in clstr if cl == 2])
    H_cl = (-1.0*ndoc1/len(clstr))*math.log(1.0*ndoc1+.0001/len(clstr),2) + (-1.0*ndoc2/len(clstr))*math.log(1.0*ndoc2+.0001/len(clstr),2)
    H_Omega += (H_cl * len(clstr)/flat_clstrs_len)
  
  Entropy.append(H_Omega)

#
#### Calc Purity
#
import math
Purity=[]
for clutering in Allclustersdist:
  pur=0
  flat_clstrs_len = len([item for sublist in clutering for item in sublist])
  if flat_clstrs_len==0:
    Purity.append(-1)
    continue
  
  for clstr in clutering:
    ndoc1= sum([True for cl in clstr if cl == 1])
    ndoc2= sum([True for cl in clstr if cl == 2])
    pur += max(ndoc1,ndoc2)
  
  Purity.append(1.0*pur/flat_clstrs_len)

cnt=0
for clutering in Allclustersdist:
  flat_clstrs_len = len([item for sublist in clutering for item in sublist])
  if flat_clstrs_len==0:
    cnt+=1

print cnt
#
#### Accuracy
#
minh=0.2
maxh=0.8
NEh=0.5
TP=FP=FN=TN=FP1=TP1=FN1=TN1=0
FNs=[]
FPs=[]
for i,z in enumerate(zip(AllNEdist,Allpureclustersratio,Allisdup_labels)):
  #if preddbscan1[i][0]>preddbscan1[i][1]:
  #  continue
  NEp,p,l=z
  if p<minh:
    if l==1:
      TP+=1
    elif l==0:
      FP+=1
      FPs.append(i)
  elif p>maxh:
    if l==0:
      TN+=1
    elif l==1:
      FN+=1
      FNs.append(i)
  else:
    if NEp>NEh:
      if l==0:
        TN1+=1
      elif l==1:
        FN1+=1
        FNs.append(i)
    else:
      if l==1:
        TP1+=1
      elif l==0:
        FP1+=1
        FPs.append(i)

Precision=1.0*TP/(TP+FP)
Recall=1.0*TP/(TP+FN)
F1=(2.0*TP)/(2.0*TP+1.0*FN+FP)
F2=(5.0*TP)/(5.0*TP+4.0*FN+FP)
print("TP",TP,"TN",TN,"FP",FP,"FN",FN)
print("\n Accuracy " + str(100.0*(TP+TN)/(TP+TN+FP+FN)) + "\n F1 " + str(F1)+"\n F2 "+ str(F2)+"\n Precision "+ str(Precision)+"\n Recall "+ str(Recall)+ str("\n TPR ")+ str(100.0*TP/(TP+FN))+ str("\n TNR ")+ str(100.0*TN/(TN+FP))+ str("\n FPR ")+ str( 100.0*FP/(TN+FP))+ str("\n FNR ")+ str( 100.0*FN/(TP+FN))+"\n n_pos "+str((TP+FN))+"\n n_neg "+str((TN+FP))+"\n positive ratio "+str((1.0*TP+FN)/(TN+FP)))
#
#
#
h=0.1
TP=FP=FN=TN=0
for i,z in enumerate(zip(AllNEdist,Allpureclustersratio,Allisdup_labels)):
  #if preddbscan1[i][0]>preddbscan1[i][1]:
  #  continue
  NEp,p,l=z
  if p<h:
    if l==1:
      TP+=1
    elif l==0:
      FP+=1
      FPs.append(i)
  else:
    if l==0:
      TN+=1
    elif l==1:
      FN+=1
      FNs.append(i)

Precision=1.0*TP/(TP+FP)
Recall=1.0*TP/(TP+FN)
F1=(2.0*TP)/(2.0*TP+1.0*FN+FP)
F2=(5.0*TP)/(5.0*TP+4.0*FN+FP)
print("TP",TP,"TN",TN,"FP",FP,"FN",FN)
print("\n Accuracy " + str(100.0*(TP+TN)/(TP+TN+FP+FN)) + "\n F1 " + str(F1)+"\n F2 "+ str(F2)+"\n Precision "+ str(Precision)+"\n Recall "+ str(Recall)+ str("\n TPR ")+ str(100.0*TP/(TP+FN))+ str("\n TNR ")+ str(100.0*TN/(TN+FP))+ str("\n FPR ")+ str( 100.0*FP/(TN+FP))+ str("\n FNR ")+ str( 100.0*FN/(TP+FN))+"\n n_pos "+str((TP+FN))+"\n n_neg "+str((TN+FP))+"\n positive ratio "+str((1.0*TP+FN)/(TN+FP)))

'''
 Accuracy 19.8072005048
 F1 0.264777753249
 F2 0.469950818786
 Precision 0.15325977757
 Recall 0.972161663592
 TPR 97.2161663592
 TNR 6.30345110047
 FPR 93.6965488995
 FNR 2.78383364078
 n_pos 17889
 n_neg 102547
 positive ratio 0.17444683901

 Accuracy 67.7629612408
 F1 0.39625546208
 F2 0.539991862614
 Precision 0.274484036365
 Recall 0.712225389904
 TPR 71.2225389904
 TNR 67.1594488381
 FPR 32.8405511619
 FNR 28.7774610096
 n_pos 17889
 n_neg 102547
 ratio 0.17444683901

 Accuracy 59.4871794872
 F1 0.23436123348
 F2 0.392099056604
 TPR 71.1229946524
 TNR 58.3758937692
 FPR 41.6241062308
 FNR 28.8770053476

 Accuracy 71.4412163201
 F1 0.357429167097
 F2 0.492484344846
 TPR 65.8314451275
 TNR 72.2109334461
 FPR 27.7890665539
 FNR 34.1685548725

 Accuracy 65.5164682646
 F1 0.222778352624
 F2 0.377071080628
 TPR 70.0514527845
 TNR 65.1722434727
 FPR 34.8277565273
 FNR 29.9485472155
 
 Accuracy 58.5608284845
 F1 0.19523118391
 F2 0.345901783878
 TPR 71.2469733656
 TNR 57.5978956316
 FPR 42.4021043684
 FNR 28.7530266344

'''
#
#### Histograms
#
#Allpureclustersratio
#Entropy
#Purity
#AllNEdist
#Allcosdist
pos=[]
neg=[]
for i,z in enumerate(zip(AllNEdist,Allpureclustersratio,Allisdup_labels)):
  p,p2,l=z
  if l==1:
    pos.append(p2)
  elif l==0:
    neg.append(p2)

plt.clf()
plt.close()
plt.cla()
f, axarr = plt.subplots(2, sharex=True)
axarr[0].hist(neg, weights=np.zeros_like(neg) + 1. / len(neg), color="blue")
axarr[0].set_title("Neg (purity ratio)")
axarr[1].hist(pos, weights=np.zeros_like(pos) + 1. / len(pos), color="red")
axarr[1].set_title("Pos (purity ratio)")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/dbscanPurityRatiohistAttro.pdf")


plt.clf()
plt.close()
plt.cla()
plt.hist(neg, weights=np.zeros_like(neg) + 1. / len(neg), color="blue")
plt.title("Neg (purityRatio)")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/negdbscanPurityRatiohistAttro.pdf")

plt.clf()
plt.close()
plt.cla()
plt.hist(pos, weights=np.zeros_like(pos) + 1. / len(pos), color="red")
plt.title("Pos (purityRatio)")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/posdbscanPurityRatiohistAttro.pdf")


#
#### Scatter Plots
#

i=1
print Allnumclusters[i],Allpureclustersratio[i],Allisdup_labels[i]
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X = tsne.fit_transform(np.vstack((w2vpairs[i][0],w2vpairs[i][1])))
Y=[1]*w2vpairs[i][0].shape[0]+[2]*w2vpairs[i][1].shape[0]
plt.clf()
plt.close()
plt.cla()
plt.scatter(X[:,0],X[:,1],c=Y)
plt.title("Words distribution of 2 Docs (not Duplicate Example - 2 pure clusters and 7 impure)")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/i"+str(i)+".pdf")

plt.clf()
plt.close()
plt.cla()
plt.scatter(X[:,0],X[:,1],c=Y,s=5)
for _ii, txt in enumerate(wordpairs[i][0]):
  plt.annotate(txt, (X[_ii,0],X[_ii,1]),fontsize=5)

for _ii, txt in enumerate(wordpairs[i][1]):
  plt.annotate(txt, (X[_ii+len(wordpairs[i][0]),0],X[_ii+len(wordpairs[i][0]),1]),fontsize=5)

plt.title("Words distribution of 2 notDup Docs")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/i"+str(i)+"words.pdf")

print Allnumclusters[i],Allpureclustersratio[i],Allisdup_labels[i]

#
#### DBscan Results
#
pridx=0
X=np.vstack((w2vpairs[pridx][0],w2vpairs[pridx][1]))
Y=[1]*w2vpairs[pridx][0].shape[0]+[2]*w2vpairs[pridx][1].shape[0]
db = DBSCAN(eps=0.5, min_samples=5).fit(X)
ll=np.unique(db.labels_)
n_pure_cl=0
n_mix_cl=0
n_noise_cl=0
pure_clstrs=[]
mixd_clstrs=[]
for _ll in ll:
  if _ll == -1:
    n_noise_cl+=1
    continue
  
  idx=np.where(db.labels_==_ll)[0]
  if len(set([Y[_idx] for _idx in idx]))>1:
    n_mix_cl+=1
    mixd_clstrs1=[]
    for _i in idx:
      if _i < w2vpairs[pridx][0].shape[0]:
        mixd_clstrs1.append(wordpairs[pridx][0][_i])
      else:
        mixd_clstrs1.append(wordpairs[pridx][1][_i-w2vpairs[pridx][0].shape[0]])
    
    mixd_clstrs.append(mixd_clstrs1)
  else:
    n_pure_cl+=1
    pure_clstrs1=[]
    for _i in idx:
      if _i < w2vpairs[pridx][0].shape[0]:
        pure_clstrs1.append(wordpairs[pridx][0][_i])
      else:
        pure_clstrs1.append(wordpairs[pridx][1][_i-w2vpairs[pridx][0].shape[0]])
    
    pure_clstrs.append(pure_clstrs1)

print 1.0*n_pure_cl/(n_pure_cl+n_mix_cl+0.00001)
print n_pure_cl,n_mix_cl
print isdup_labels[pridx]
for cl in pure_clstrs:
  print cl

for cl in mixd_clstrs:
  print cl

#
#### Scatter Plots
#

i=1
print Allnumclusters[i],Allpureclustersratio[i],Allisdup_labels[i]
#AllNEdist,Allpureclustersratio,Allisdup_labels
plt.scatter(Allpureclustersratio,AllNEdist,c=Allisdup_labels)
plt.title("NEdist vs. pureclustersratio (color is label)")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/NEvsClstr.pdf")

plt.clf()
plt.close()
plt.cla()
plt.scatter(X[:,0],X[:,1],c=Y,s=5)
for _ii, txt in enumerate(wordpairs[i][0]):
  plt.annotate(txt, (X[_ii,0],X[_ii,1]),fontsize=5)

for _ii, txt in enumerate(wordpairs[i][1]):
  plt.annotate(txt, (X[_ii+len(wordpairs[i][0]),0],X[_ii+len(wordpairs[i][0]),1]),fontsize=5)

plt.title("Words distribution of 2 notDup Docs")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/i"+str(i)+"words.pdf")

print Allnumclusters[i],Allpureclustersratio[i],Allisdup_labels[i]


