from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import cPickle
import matplotlib.pyplot as plt
from multiprocessing import Pool,cpu_count
from sklearn.preprocessing import StandardScaler
#import hdbscan


def dbclustering_purity(_w2vpairs,dbscan_eps=0.5, dbscan_minPts=2,min_samples_pt=2):
  
  
  #_w2vpairs=_w2vpr
  #_wordspairs=wordsprs
  
  if _w2vpairs[0].size ==0 or _w2vpairs[1].size ==0:
    return [[],[-100000,-100000,-100000], -100000]
  
  
  X=np.vstack((_w2vpairs[0],_w2vpairs[1]))
  X = StandardScaler().fit_transform(X)
  Y=[1]*_w2vpairs[0].shape[0]+[2]*_w2vpairs[1].shape[0]
  
  distance = cosine_similarity(X)+1
  distance = distance/np.max(distance)
  distance = 1 - distance
  db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, metric='precomputed', n_jobs=1).fit(distance.astype('float64'))
  #db = hdbscan.HDBSCAN(min_samples = min_samples_pt, min_cluster_size=dbscan_minPts, metric='precomputed').fit(distance.astype('float64'))
  #print distance
  def cos_metric(x, y):
    i, j = int(x[0]), int(y[0])# extract indices
    #print cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
    return cosine_similarity(X[i,].reshape(1,-1),X[j,].reshape(1,-1))
  
  #db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, n_jobs=1, metric=cos_metric).fit(X)
  #db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minPts, n_jobs=-1).fit(X)
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

run(h=0.40)
def run(h=0.335):
  posclustersdist=[]
  posnumclusters=[]
  pospureclustersratio=[]
  for _w2vpr in embeddingspair:
      clustersdist,numclusters,pureclustersratio=dbclustering_purity(_w2vpr,dbscan_eps=h, dbscan_minPts=2, min_samples_pt =2)
      posclustersdist.append(clustersdist)
      posnumclusters.append(numclusters)
      pospureclustersratio.append(pureclustersratio)
  
  print len(posclustersdist),len(posnumclusters),len(pospureclustersratio),len(embeddingspair)
  
  #print posclustersdist[:100]
  #print posnumclusters[:100]
  print np.percentile(pospureclustersratio,[10,20,50,80,90])
  
  
  negclustersdist=[]
  negnumclusters=[]
  negpureclustersratio=[]
  for _i1,_i2 in negativeidx:
      _w2vpr=[embeddingspair[_i1][0],embeddingspair[_i2][1]]
      #_wordspairs=[wordspair[_i1][0],wordspair[_i2][1]]
      clustersdist,numclusters,pureclustersratio=dbclustering_purity(_w2vpr,dbscan_eps=h, dbscan_minPts=2, min_samples_pt =2)
      negclustersdist.append(clustersdist)
      negnumclusters.append(numclusters)
      negpureclustersratio.append(pureclustersratio)
  
  #print negclustersdist[:100]
  #print negnumclusters[:100]
  #print np.percentile(pospureclustersratio,[10,20,50,80,90])
  print np.percentile(negpureclustersratio,[10,20,50,80,90])
  
  lg=['en','ar']
  TP=sum([True for p in pospureclustersratio if p<=0.5])
  FP=sum([True for p in negpureclustersratio if p<=0.5])
  TN=sum([True for p in negpureclustersratio if p>0.5])
  FN=sum([True for p in pospureclustersratio if p>0.5])
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
  F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
  print("TP",TP,"TN",TN,"FP",FP,"FN",FN)
  print("\n Accuracy " + str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "\n F1 " + str(F1)+"\n F2 "+ str(F2)+"\n Precision "+ str(Precision)+"\n Recall "+ str(Recall)+ str("\n TPR ")+ str(100.0*TP/(TP+FN+0.0001))+ str("\n TNR ")+ str(100.0*TN/(TN+FP+0.0001))+ str("\n FPR ")+ str( 100.0*FP/(TN+FP+0.0001))+ str("\n FNR ")+ str( 100.0*FN/(TP+FN+0.0001))+"\n n_pos "+str((TP+FN))+"\n n_neg "+str((TN+FP))+"\n positive ratio "+str((1.0*TP+FN)/(TN+FP)))
  print h,lg[0],lg[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))



run(h=0.388)



def runEvReg(h=0.4,ln0='es',ln1='en'):
  embeddingspair=[ w2vpr for w2vpr,lnpr in zip(w2vpairsList,Alllangpairs) if lnpr[0]==ln0 and lnpr[1]==ln1]
  isdup_labels=[ lbl for lbl,lnpr in zip(Allisdup_labels,Alllangpairs) if lnpr[0]==ln0 and lnpr[1]==ln1]
  Allclustersdist=[]
  Allnumclusters=[]
  Allpureclustersratio=[]
  for _embeddingpr in embeddingspair:
      clustersdist,numclusters,pureclustersratio=dbclustering_purity(_embeddingpr,dbscan_eps=h, dbscan_minPts=2, min_samples_pt =2)
      #Allclustersdist.append(clustersdist)
      #Allnumclusters.append(numclusters)
      Allpureclustersratio.append(pureclustersratio)
  
  #print len(Allclustersdist),len(Allnumclusters),len(Allpureclustersratio),len(embeddingspair),len(isdup_labels)
  lg=[ln0,ln1]
  TP=sum([True for p,l in zip(Allpureclustersratio,isdup_labels) if p<=0.5 and l==1])
  FP=sum([True for p,l in zip(Allpureclustersratio,isdup_labels) if p<=0.5 and l==0])
  TN=sum([True for p,l in zip(Allpureclustersratio,isdup_labels) if p>0.5 and l==0])
  FN=sum([True for p,l in zip(Allpureclustersratio,isdup_labels) if p>0.5 and l==1])
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
  F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
  #print("TP",TP,"TN",TN,"FP",FP,"FN",FN)
  #print("\n Accuracy " + str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "\n F1 " + str(F1)+"\n F2 "+ str(F2)+"\n Precision "+ str(Precision)+"\n Recall "+ str(Recall)+ str("\n TPR ")+ str(100.0*TP/(TP+FN+0.0001))+ str("\n TNR ")+ str(100.0*TN/(TN+FP+0.0001))+ str("\n FPR ")+ str( 100.0*FP/(TN+FP+0.0001))+ str("\n FNR ")+ str( 100.0*FN/(TP+FN+0.0001))+"\n n_pos "+str((TP+FN))+"\n n_neg "+str((TN+FP))+"\n positive ratio "+str((1.0*TP+FN)/(TN+FP)))
  print h,ln0,ln1,TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001))


for _h in [0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42]:
  runEvReg(h=_h,ln0='es',ln1='de')

for _h in [0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42]:
  runEvReg(h=_h,ln0='de',ln1='es')



runEvReg(h=0.39,ln0='en',ln1='de')
runEvReg(h=0.40,ln0='en',ln1='de')


