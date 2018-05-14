from sklearn.metrics.pairwise import cosine_similarity



unifiedw2vmodel=loadunifiedw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
#unifiedw2vmodel=loadunifiedw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/twelve.table4.multiSkip.size_512+w_5+it_10.normalized')

Allwordspairs=[]
w2vpairsList=[]
Alllangpairs=[]
Allisdup_labels=[]
posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
allposfiles = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
allnegfiles = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]
to=min(len(allposfiles),len(allnegfiles))
print to
frm=0
cnt=0
for frm in range(0,to-50,50):
  w2vpairs,labels,wordspairs,langpairs,EntJaccardSim,sentpairs=create_w2v_pairs(unifiedw2vmodel,allposfiles[frm:frm+10],allnegfiles[frm:frm+10])
  print "processing ",frm,len(w2vpairs),len(w2vpairs[0]), " pairs"
  if len(w2vpairs)==0:
    continue
  
  w2vpairsList.extend(w2vpairs)
  Allwordspairs.extend(wordspairs)
  Allisdup_labels.extend(labels)
  Alllangpairs.extend(langpairs)

#
#

w2vlist1=[]
w2vlist2=[]
Allisdup_lbls=[]
Alllangprs=[]
for w2vpr,isdup,lang in zip(w2vpairsList,Allisdup_labels,Alllangpairs):
  if w2vpr[0].shape[0]>0:
    w2vlist1.append(np.mean(w2vpr[0],axis=0))
    w2vlist2.append(np.mean(w2vpr[1],axis=0))
    Allisdup_lbls.append(isdup)
    Alllangprs.append(lang)


w2vlist=np.array(w2vlist1+w2vlist2)
X = StandardScaler().fit_transform(w2vlist)

a=cosine_similarity(X[:len(w2vlist1),],X[len(w2vlist1):,])


cossim=[]
for l1 in range(len(w2vlist1)):
  cossim.append(cosine_similarity(X[l1,].reshape(1,-1),X[l1+len(w2vlist1),].reshape(1,-1)))

cossim=[sim[0][0] for sim in cossim]

cnt=0
langposneg=dict()
for i,z in enumerate(zip(cossim,Allisdup_lbls)):
  p,l=z
  langpairstr=str(Alllangprs[i])
  if langpairstr not in langposneg.keys():
    langposneg[langpairstr]=[[],[]]
  
  if p<0:
    cnt+=1
  
  langposneg[langpairstr][l].append(p)



for _langposneg in langposneg.keys():
  pos=langposneg[_langposneg][1]
  neg=langposneg[_langposneg][0]
  print _langposneg,len(pos),len(neg)



tp=tn=fp=fn=0
for lg in langposneg.keys():
  lg=lg[1:-1].strip().split(",")
  lg=[lg[0].strip()[1:-1],lg[1].strip()[1:-1]]
  P=[pp for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]]
  N=[pp for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]]
  pvalue=P+N
  lbls=[1]*len(P)+[0]*len(N)
  argssorted=np.argsort(pvalue)
  pvaluesorted=[pvalue[l] for l in argssorted ]
  lblssorted=[lbls[l] for l in argssorted ]
  maxF1=0
  besth=-1
  for _idx,h in enumerate(pvaluesorted):
    TP=sum(lbls[:_idx])
    FP=_idx-TP
    FN=sum(lbls[_idx:])
    F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
    if maxF1 < F1:
      maxF1=F1
      besth=h
  
  h=besth
  TP=sum([True for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if pp<=h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  FP=sum([True for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if pp<=h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  TN=sum([True for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if pp>h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  FN=sum([True for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if pp>h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  poserror=sum([True for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if pp<0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  negerror=sum([True for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if pp<0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  tp+=TP
  tn+=TN
  fp+=FP
  fn+=FN
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
  F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
  print lg[0],lg[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001)),h,poserror,negerror


precision=100.0*tp/(tp+fp+0.0001)
recall=100.0*tp/(tp+fn+0.0001)
f1=100.0*(2.0*tp)/((2.0*tp+1.0*fn+fp)+0.000001)
f2=100.0*(5.0*tp)/((5.0*tp+4.0*fn+fp)+0.000001)
print tp,tn,fp,fn,str(100.0*(tp+tn)/(tp+tn+fp+fn++0.000001)) + "," + str(f1)+", "+ str(f2)+", "+ str(precision)+", "+ str(recall)+ str(", ")+ str(100.0*tp/(tp+fn+0.0001))+ str(", ")+ str(100.0*tn/(tn+fp+0.0001))+ str(", ")+ str( 100.0*fp/(tn+fp+0.0001))+ str(", ")+ str( 100.0*fn/(tp+fn+0.0001))+", "+str((tp+fn))+", "+str((tn+fp))+", "+str((1.0*tp+fn)/(tn+fp+tp+fn+0.0001))



tp=tn=fp=fn=0
for lg in langposneg.keys():
  lg=lg[1:-1].strip().split(",")
  lg=[lg[0].strip()[1:-1],lg[1].strip()[1:-1]]
  h=0.5
  TP=sum([True for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if pp<=h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  FP=sum([True for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if pp<=h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  TN=sum([True for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if pp>h and pp>=0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  FN=sum([True for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if pp>h and pp>=0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  poserror=sum([True for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if pp<0 and lbl==1 and lng[0]==lg[0] and lng[1]==lg[1]])
  negerror=sum([True for pp,lbl,lng in zip(cossim,Allisdup_lbls,Alllangprs) if pp<0 and lbl==0 and lng[0]==lg[0] and lng[1]==lg[1]])
  tp+=TP
  tn+=TN
  fp+=FP
  fn+=FN
  Precision=100.0*TP/(TP+FP+0.000001)
  Recall=100.0*TP/(TP+FN+0.000001)
  F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
  F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
  print lg[0],lg[1],TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001)),h,poserror,negerror


precision=100.0*tp/(tp+fp+0.0001)
recall=100.0*tp/(tp+fn+0.0001)
f1=100.0*(2.0*tp)/((2.0*tp+1.0*fn+fp)+0.000001)
f2=100.0*(5.0*tp)/((5.0*tp+4.0*fn+fp)+0.000001)
print tp,tn,fp,fn,str(100.0*(tp+tn)/(tp+tn+fp+fn++0.000001)) + "," + str(f1)+", "+ str(f2)+", "+ str(precision)+", "+ str(recall)+ str(", ")+ str(100.0*tp/(tp+fn+0.0001))+ str(", ")+ str(100.0*tn/(tn+fp+0.0001))+ str(", ")+ str( 100.0*fp/(tn+fp+0.0001))+ str(", ")+ str( 100.0*fn/(tp+fn+0.0001))+", "+str((tp+fn))+", "+str((tn+fp))+", "+str((1.0*tp+fn)/(tn+fp+tp+fn+0.0001))
