import cPickle

Allnumclusters=cPickle.load(open('/home/ahmad/duplicate-detection/atrocitiesdata/AllnumclustersAttro2.p', 'rb'))
Allpureclustersratio=cPickle.load(open('/home/ahmad/duplicate-detection/atrocitiesdata/AllpureclustersratioAttro2.p', 'rb'))
Allcosdist=cPickle.load(open('/home/ahmad/duplicate-detection/atrocitiesdata/AllcosdistAttro2.p', 'rb'))
AllNEdist=cPickle.load(open('/home/ahmad/duplicate-detection/atrocitiesdata/AllNEdistAttro2.p', 'rb'))

Allclustersdist=cPickle.load(open('/home/ahmad/duplicate-detection/atrocitiesdata/AllclustersdistAttro2.p', 'rb'))
Allisdup_labels=cPickle.load(open('/home/ahmad/duplicate-detection/atrocitiesdata/Allisdup_labelsAttro2.p', 'rb'))


cosd=[]
eucd=[]
dataset=[]

duplabels=[]
pratio=[]
langpr=[]
for clstring,lbl,_langpr in zip(Allclustersdist,Allisdup_labels,Alllangpairs):
  if len(clstring)==0:
    continue
  instance=[]
  instanceA=[]
  instanceB=[]
  _eucd=0
  for clstr in clstring:
    n_doc1=0
    n_doc2=0
    for doc in clstr:
      if doc==1:
        n_doc1+=1
      elif doc==2:
        n_doc2+=1
    
    instance.append(1.0*abs(n_doc1-n_doc2)/max(n_doc1,n_doc2))
    _eucd+=pow(n_doc1-n_doc2,2)
    instanceA.append(n_doc1)
    instanceB.append(n_doc2)
  
  if len(instance)>0:
    pratio.append(1.0*sum([True for ones in instance if ones == 1.0])/len(instance))
    langpr.append(_langpr)
    duplabels.append(lbl)

    eucd.append(pow(_eucd,0.5))
    cosd.append(spatial.distance.cosine(instanceA,instanceB))
    dataset.append(instance)


print len(dataset),len(duplabels),len(pratio),len(cosd),len(loc),len(ne)
print len(Allclustersdist),len(Allisdup_labels),len(Alllangpairs)

dataset=[]
duplabels=[]
pratio=[]
cosd=[]
eucd=[]
langpr=[]
loc=[]
ne=[]
#for clstring,lbl,_langpr in zip(Allclustersdist,Allisdup_labels,Alllangpairs):
for clstring,lbl,_loc,_ne in zip(Allclustersdist,Allisdup_labels,Locdist,NEdist):
  if len(clstring)==0:
    continue
  instance=[]
  instanceA=[]
  instanceB=[]
  _eucd=0
  for clstr in clstring.values():
    n_doc1=0
    n_doc2=0
    for doc in clstr:
      if doc==1:
        n_doc1+=1
      elif doc==2:
        n_doc2+=1
    
    instance.append(1.0*abs(n_doc1-n_doc2)/max(n_doc1,n_doc2))
    _eucd+=pow(n_doc1-n_doc2,2)
    instanceA.append(n_doc1)
    instanceB.append(n_doc2)
  
  if len(instance)>0:
    pratio.append(1.0*sum([True for ones in instance if ones == 1.0])/len(instance))
    eucd.append(pow(_eucd,0.5))
    cosd.append(spatial.distance.cosine(instanceA,instanceB))
    dataset.append(instance)
    #langpr.append(_langpr)
    duplabels.append(lbl)
    loc.append(_loc)
    ne.append(_ne)

print len(dataset),len(duplabels),len(pratio),len(cosd),len(loc),len(ne)

cnt=0
for clstring,lbl in zip(Allclustersdist,Allisdup_labels):
  if len(clstring)==0:
    cnt+=1

min([len(dd) for dd in dataset])
a=[len(dd) for dd in dataset]
leng=dict()
for ii in range(len(a)):
  
  strstr=str(a[ii])
  if strstr not in leng.keys():
    leng[strstr]=0
  
  leng[strstr]+=1

b=leng.values()
b.sorted()
max(leng.values())
print lang

from sklearn.svm import SVC
X=[]
y=[]
unsupX=[]
for dd,_y,_unsupX in zip(dataset,duplabels,pratio):
  if len(dd) == 5:
    X.append(dd)
    unsupX.append(_unsupX)
    y.append(_y)

clf = SVC()
X=np.asarray(X)
ntran=2*X.shape[0]/3
clf.fit(X[:ntran,:], y[:ntran])
pred=clf.predict(X[ntran:,:])
TP=FP=FN=TN=0
for i,z in enumerate(zip(pred,y[ntran:])):
  #for i,z in enumerate(zip(unsupX[ntran:],y[ntran:]):
  p,l=z
  if p==1:
    if l==1:
      TP+=1
    elif l==0:
      FP+=1
  else:
    if l==0:
      TN+=1
    elif l==1:
      FN+=1

#
#
### Scatter plot
#
#

from sklearn import manifold
from nltk.corpus import stopwords
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
w2vpairs,labels,wordspairs,langpairs,EntJaccardSim

i=langpairs.index(['en', 'es'])
i=0
a=np.sum(Allnumclusters,axis=0)
i=np.where(a==5)[0][0]

print Allpureclustersratio[i]
print Allisdup_labels[i]
print Allnumclusters[i]
print Allclustersdist[i]
en_stop = set(stopwords.words('spanish'))
wordslist=set(wordspairs[i][1])
w2vmatrix1=[]
wlist=[]
for word in wordslist:
  if '' !=word.strip().lower() and word.strip().lower() not in en_stop:
    try:
      w2vmatrix1.append(unifiedw2vmodel[word.strip().lower()])
      #w2vmatrix1.append(w2vmodel[langcode[artcle['lang']]][word.strip().lower()])
      wlist.append(word.strip().lower())
    except:
      pass

w2vmatrix=[]
w2vmatrix.append(np.asarray(w2vmatrix1))

wlist1=[]
wlist1.append(wlist)

X = tsne.fit_transform(np.vstack((w2vpairs[i][0],w2vpairs[i][1])))
Y=[1]*w2vpairs[i][0].shape[0]+[2]*w2vpairs[i][1].shape[0]

db = DBSCAN(eps=0.5, min_samples=5).fit(X)
ll=np.unique(db.labels_)
n_pure_cl=0
n_noise_cl=0
for _ll in ll:
  idx=np.where(db.labels_==_ll)[0]
  if len(set([Y[_idx] for _idx in idx]))>1:
    n_noise_cl+=1
  else:
    n_pure_cl+=1

print 1.0*n_pure_cl/(n_noise_cl+n_pure_cl)
plt.clf()
plt.close()
plt.cla()
plt.scatter(X[:w2vpairs[i][0].shape[0],0],X[:w2vpairs[i][0].shape[0],1],c="blue",marker="x")
plt.scatter(X[w2vpairs[i][0].shape[0]:,0],X[w2vpairs[i][0].shape[0]:,1],c="red",marker="o") #,s=5

plt.scatter(X[:w2vpairs[i][0].shape[0],0],X[:w2vpairs[i][0].shape[0],1],c=db.labels_[:w2vpairs[i][0].shape[0]],marker="x")
plt.scatter(X[w2vpairs[i][0].shape[0]:,0],X[w2vpairs[i][0].shape[0]:,1],c=db.labels_[w2vpairs[i][0].shape[0]:],marker="o") #,s=5

for _ii, txt in enumerate(wordspairs[i][0]):
  plt.annotate(txt, (X[_ii,0],X[_ii,1]),fontsize=5)

for _ii, txt in enumerate(wordspairs[i][1]):
  plt.annotate(txt, (X[_ii+len(wordspairs[i][0]),0],X[_ii+len(wordspairs[i][0]),1]),fontsize=5)

#plt.title("Words distribution of 2 notDup Docs")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/nonduplicateexample.pdf")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/i"+str(i)+"words.pdf")

'''
plt.clf()
plt.close()
plt.cla()
plt.scatter(X[:,0],X[:,1],c=Y)
plt.savefig("i"+str(i)+".pdf")

plt.clf()
plt.close()
plt.cla()
plt.scatter(X[:,0],X[:,1],c=Y,s=5)
for _ii, txt in enumerate(wordpairs[i][0]):
  plt.annotate(txt, (X[_ii,0],X[_ii,1]),fontsize=5)

for _ii, txt in enumerate(wordpairs[i][1]):
  plt.annotate(txt, (X[_ii,0],X[_ii,1]),fontsize=5)

plt.title("Docs distribution (FN)")
plt.savefig("i"+str(i)+"words.pdf")
'''

'''
8 clusters
Supervised
  Accuracy 92.2673656619
 F1 57.2463768116
 F2 45.5594002307
 Precision 100.0
 Recall 40.1015228426
 TPR 40.1015228426
 TNR 100.0
 FPR 0.0
 FNR 59.8984771574
 n_pos 197
 n_neg 1329
 positive ratio 0.148231753198

UnSupervised
  Accuracy 88.9252948886
 F1 61.8510158014
 F2 66.247582205
 Precision 55.6910569106
 Recall 69.5431472081
 TPR 69.5431472081
 TNR 91.79834462
 FPR 8.20165537998
 FNR 30.4568527919
 n_pos 197
 n_neg 1329
 positive ratio 0.148231753198
'''

a

'''
print len(Allcosdist),len(plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/dbscanSetwiseCosdhistAttro2.pdf")),len(AllPersondist),len(AllOrgsdist),len(AllDatedist),len(Allpureclustersratio),len(Allisdup_labels)
pos=[]
neg=[]
for i,z in enumerate(zip(AllNEdist,Allisdup_labels)):
  #for i,z in enumerate(zip(unsupX,y)):
  p2,l=z
  if np.isnan(p2):
    continue
  if l==1:
    pos.append(p2)
  elif l==0:
    neg.append(p2)

plt.clf()
plt.close()
plt.cla()
f, axarr = plt.subplots(2, sharex=True)
axarr[0].hist(neg, weights=np.zeros_like(neg) + 1. / len(neg), color="blue")
axarr[0].set_title("Negative (Named Entities distance)")
axarr[1].hist(pos, weights=np.zeros_like(pos) + 1. / len(pos), color="red")
axarr[1].set_title("Positive (Named Entities distance)")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/AllNEdistthistAttro2.pdf")
plt.savefig("/home/ahmad/duplicate-detection/eventregistrydata/dbscanSetwiseCosdhistAttro2.pdf")



h=0.2
TP=FP=FN=TN=0
#for i,z in enumerate(zip(dataset,pratio,duplabels)):
for i,z in enumerate(zip(unsupX[ntran:],y[ntran:])):
  #if preddbscan1[i][0]>preddbscan1[i][1]:
  #  continue
  p,l=z
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

Precision=100.0*TP/(TP+FP)
Recall=100.0*TP/(TP+FN)
F1=100.0*(2.0*TP)/(2.0*TP+1.0*FN+FP)
F2=100.0*(5.0*TP)/(5.0*TP+4.0*FN+FP)
print("TP",TP,"TN",TN,"FP",FP,"FN",FN)
print("\n Accuracy " + str(100.0*(TP+TN)/(TP+TN+FP+FN)) + "\n F1 " + str(F1)+"\n F2 "+ str(F2)+"\n Precision "+ str(Precision)+"\n Recall "+ str(Recall)+ str("\n TPR ")+ str(100.0*TP/(TP+FN))+ str("\n TNR ")+ str(100.0*TN/(TN+FP))+ str("\n FPR ")+ str( 100.0*FP/(TN+FP))+ str("\n FNR ")+ str( 100.0*FN/(TP+FN))+"\n n_pos "+str((TP+FN))+"\n n_neg "+str((TN+FP))+"\n positive ratio "+str((1.0*TP+FN)/(TN+FP)))

 Attrocities -50%-
h=0.1
 Accuracy 76.2600891206
 F1 43.418580768
 F2 59.3305038335
 Precision 30.0062318238
 Recall 78.5125464263
 TPR 78.5125464263
 TNR 75.9644763592
 FPR 24.0355236408
 FNR 21.4874535737
 n_pos 11039
 n_neg 84113
 positive ratio 0.131240117461
-------------------------------
h=0.2
 Accuracy 77.3278543804
 F1 0.425133903589
 F2 0.564591472737
 Precision 0.301155240109
 Recall 0.722619802518
 TPR 72.2619802518
 TNR 77.992700296
 FPR 22.007299704
 FNR 27.7380197482
 n_pos 11039
 n_neg 84113
 positive ratio 0.131240117461

'''
