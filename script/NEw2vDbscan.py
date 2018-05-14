len(transallleftNE),len(transallrightNE),len(lbl),len(lng)


def dbclustering_purity(_w2vpairs,dbscan_eps=0.5, dbscan_minPts=2,min_samples_pt=2):
  
  if _w2vpairs[0].size ==0 or _w2vpairs[1].size ==0:
    return [[],[-100000,-100000,-100000], -100000]
  
  
  X=np.vstack((_w2vpairs[0],_w2vpairs[1]))
  X = StandardScaler().fit_transform(X)
  Y=[1]*_w2vpairs[0].shape[0]+[2]*_w2vpairs[1].shape[0]
  
  distance = cosine_similarity(X)+1
  distance = distance/np.max(distance)
  distance = 1 - distance
  db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, metric='precomputed', n_jobs=1).fit(distance.astype('float64'))
  
  def cos_metric(x, y):
    i, j = int(x[0]), int(y[0])# extract indices
    #print cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
    return cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
  
  labels_=list(db.labels_)
  #labels_=dbscan(X, eps=0.5, min_samples=5)[1]
  _n=len(set(labels_))
  if -1 in labels_:
    _n -= 1
  clusters= [[] for _ in range(_n)]
  n_pure_cl=0
  n_noise_cl=0
  n_mixed_cl=0
  n_pure_1=0
  n_pure_2=0
  for _idx,_lbl in enumerate(labels_):
    if _lbl==-1:
      n_noise_cl+=1
    else:
      clusters[_lbl].append(Y[_idx])
  
  for _lbl in clusters:
    if len(set(_lbl))>1:
      n_mixed_cl+=1
    else:
      n_pure_cl+=1
      if _lbl[0]==1:
        n_pure_1+=1
      elif _lbl[0]==2:
        n_pure_2+=1
  
  #print n_pure_1,n_pure_2,n_mixed_cl
  if min(n_pure_1+n_mixed_cl,n_pure_2+n_mixed_cl)==0:
    return [clusters, [n_pure_cl,n_mixed_cl,n_noise_cl], 1.0]
  else:
    return [clusters, [n_pure_cl,n_mixed_cl,n_noise_cl], 1.0*min(n_pure_1,n_pure_2)/(min(n_pure_1,n_pure_2)+n_mixed_cl+0.00001)]


sum([True for _lng in lng if 'es' == _lng[0] and 'en' == _lng[1]])
sum([True for _score in score if _score>1])
print len(transallleftNE),len(transallrightNE),len(lbl),len(lng)
#embeddingsmodel=loadfasttextmodel('Path To Vectors')

Allpureclustersratio=[]
labels=[]
dbh=0.1
_lng0='de'
_lng1='en'
for idx in range(len(lbl)):
  if lng[idx][0]!=_lng0 or lng[idx][1]!=_lng1:
    continue
  
  w2vmatrix1=[]
  wlist=[]
  for word in transallleftNE[idx]:
    if '' !=word.strip() and word.strip().lower() not in stpwords:
      try:
        if type(word)!=type(''):
          word=word.strip().lower().encode('utf-8')
        else:
          word=word.strip().lower()
        
        w2vmatrix1.append(list(embeddingsmodel['en'][word]))
      except:
        pass
  
  embeddingpr=[np.array(w2vmatrix1)]
  
  w2vmatrix2=[]
  for word in transallrightNE[idx]:
    if '' !=word.strip() and word.strip().lower() not in stpwords:
      try:
        if type(word)!=type(''):
          word=word.strip().lower().encode('utf-8')
        else:
          word=word.strip().lower()
        
        w2vmatrix2.append(list(embeddingsmodel['en'][word]))
      except:
        pass
  
  embeddingpr.append(np.array(w2vmatrix2))
  
  if len(embeddingpr[0])==0 or len(embeddingpr[1])==0:
    print idx
    continue
  
  
  if idx%1000==0:
    print "processing ",idx
  
  clustersdist,numclusters,pureclustersratio=dbclustering_purity(embeddingpr,dbscan_eps=dbh, dbscan_minPts=2, min_samples_pt =2)
  Allpureclustersratio.append(pureclustersratio)
  labels.append(lbl[idx])

countpos=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==1 and _lng[0]==_lng0 and _lng[1]==_lng1])
countneg=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==0 and _lng[0]==_lng0 and _lng[1]==_lng1])

h=0.5
TP=sum([True for pp,_lbl in zip(Allpureclustersratio,labels) if pp<=h and pp>=0 and _lbl==1])
FP=sum([True for pp,_lbl in zip(Allpureclustersratio,labels) if pp<=h and pp>=0 and _lbl==0])
TN=sum([True for pp,_lbl in zip(Allpureclustersratio,labels) if pp>h and pp>=0 and _lbl==0])
FN=sum([True for pp,_lbl in zip(Allpureclustersratio,labels) if pp>h and pp>=0 and _lbl==1])


d=countpos-(TP+FN)
FN+=d
d=countneg-(TN+FP)
FP+=d

poserror=sum([True for pp,_lbl in zip(Allpureclustersratio,labels) if pp<0 and _lbl==1])
negerror=sum([True for pp,_lbl in zip(Allpureclustersratio,labels) if pp<0 and _lbl==0])
Precision=100.0*TP/(TP+FP+0.000001)
Recall=100.0*TP/(TP+FN+0.000001)
F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
print dbh,'de','es',TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001)),h,poserror,negerror

print Precision,Recall
print 100.0*FP/(FP+TN),100.0*FN/(FN+TP)

