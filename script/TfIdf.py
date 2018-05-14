from nltk.corpus import stopwords
import gensim
import os
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
#from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
from datetime import datetime,date,timedelta
#import load
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import manifold
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

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


def distancepairs(tfidfpairs):
  cosdistance=[]
  for pridx in range(len(tfidfpairs)): 
    cosdistance.append(spatial.distance.cosine(tfidfpairs[pridx][0],tfidfpairs[pridx][1]))
    #cosine_similarity(pairs[pridx][0],pairs[pridx][1])
  
  return cosdistance









if __name__ == '__main__':
sentencesfile='/home/ahmad/duplicate-detection/atrocitiesdata/TextList3.txt'
dataText=get_doc_list(sentencesfile)
dataText_vectorizer = TfidfVectorizer(analyzer = "word",tokenizer = tokenize, preprocessor = None, max_features = 1000) #,ngram_range=(1, 1)
dataText_features = dataText_vectorizer.fit_transform(dataText)
dataText_features = dataText_features.toarray()
n=len(dataText)
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

Allnumclusters=[]
Allpureclustersratio=[]
Allclustersdist=[]
Allcosdist=[]
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
    Trmask=np.zeros(n,dtype='bool')
    Trmask[Trainingindexes]=True
    datax=dataText_features[Trmask,:]
    documents = [w2vdata[i] for i in Trainingindexes]
    docterms = [wordsdata[i] for i in Trainingindexes]
    clusterslabels=[ClusterLabel[i] for i in Trainingindexes]
    Tfidfpairs=[]
    w2vpairs=[]
    isdup_labels=[]
    wordpairs=[]
    for x1 in range(len(documents)):
      for x2 in range(x1,len(documents)):
        w2vpairs.append([documents[x1],documents[x2]])
        Tfidfpairs.append([datax[x1],datax[x2]])
        wordpairs.append([docterms[x1],docterms[x2]])
        if clusterslabels[x1]==clusterslabels[x2]:
          isdup_labels.append(1)
        else:
          isdup_labels.append(0)
    
    numclusters,pureclustersratio,clustersdist=clustering_purity(w2vpairs,dbscan_eps=0.5, dbscan_minPts=5)
    Allcosdist.extend(distancepairs(Tfidfpairs))
    Allclustersdist.extend(clustersdist)
    Allnumclusters.extend(numclusters)
    Allpureclustersratio.extend(pureclustersratio)
    Allisdup_labels.extend(isdup_labels)
    
    Trainingindexes=[]

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
#h=0.1
h=.95
TP=FP=FN=TN=0
FNs=[]
FPs=[]
for i,z in enumerate(zip(AllNEdist,Allisdup_labels)):
  #if preddbscan1[i][0]>preddbscan1[i][1]:
  #  continue
  p,l=z
  if p<h and l==1:
    TP+=1
  elif p<h and l==0:
    FP+=1
    FPs.append(i)
  elif p>h and l==0:
    TN+=1
  elif p>h and l==1:
    FN+=1
    FNs.append(i)

Precision=100.0*TP/(TP+FP)
Recall=100.0*TP/(TP+FN)
F1=100.0*(2.0*TP)/(2.0*TP+1.0*FN+FP)
F2=100.0*(5.0*TP)/(5.0*TP+4.0*FN+FP)
print("TP",TP,"TN",TN,"FP",FP,"FN",FN)
print("\n Accuracy " + str(100.0*(TP+TN)/(TP+TN+FP+FN)) + "\n F1 " + str(F1)+"\n F2 "+ str(F2)+"\n Precision "+ str(Precision)+"\n Recall "+ str(Recall)+ str("\n TPR ")+ str(100.0*TP/(TP+FN))+ str("\n TNR ")+ str(100.0*TN/(TN+FP))+ str("\n FPR ")+ str( 100.0*FP/(TN+FP))+ str("\n FNR ")+ str( 100.0*FN/(TP+FN))+"\n n_pos "+str((TP+FN))+"\n n_neg "+str((TN+FP))+"\n positive ratio "+str((1.0*TP+FN)/(TN+FP)))

print(TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+0.0001)))
'''
 
  TFIDF:
 Accuracy 48.4647447607
 F1 0.285016530544
 F2 0.440324325863
 Precision 0.179497968659
 Recall 0.691542288557
 TPR 69.1542288557
 TNR 44.8555296596
 FPR 55.1444703404
 FNR 30.8457711443
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
#Allcosdist
#AllLocationdist
#AllPersondist
#AllOrgsdist
#AllDatedist
pos=[]
neg=[]
for i,z in enumerate(zip(Allcosdist,Allisdup_labels)):
  p,l=z
  if l==1:
    pos.append(p)
  elif l==0:
    neg.append(p)

plt.clf()
plt.close()
plt.cla()
f, axarr = plt.subplots(2, sharex=True)
axarr[0].hist(neg, weights=np.zeros_like(neg) + 1. / len(neg), color="blue")
axarr[0].set_title("Negative (Cosine distance)")
axarr[1].hist(pos, weights=np.zeros_like(pos) + 1. / len(pos), color="red")
axarr[1].set_title("Positive (Cosine distance)")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/AllcosdisthistAttro2.pdf")

plt.clf()
plt.close()
plt.cla()
plt.hist(neg, weights=np.zeros_like(neg) + 1. / len(neg), color="blue")
plt.title("Negatives Dates")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/negdbscanAllDatedisthistAttro.pdf")

plt.clf()
plt.close()
plt.cla()
plt.hist(pos, weights=np.zeros_like(pos) + 1. / len(pos), color="red")
plt.title("Positives Dates")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/posdbscanAllDatedisthistAttro.pdf")

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
