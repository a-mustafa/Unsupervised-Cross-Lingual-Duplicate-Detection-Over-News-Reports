'''Train a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import sys
import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, merge,Flatten
from keras.optimizers import RMSprop, Adam
from keras import backend as K

import time
from os import listdir
from os.path import isfile, join
import re
import json
from sklearn.metrics.pairwise import cosine_similarity


#
#
#
#
#

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def minmax_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1, 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

'''
def create_pairs(x, digit_indices):
    #Positive and negative pair creation.
    #Alternates between positive and negative pairs.
    
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)
'''

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

'''
# the data, shuffled and split between train and test sets
w2vmodel= loadmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
breaks=CMF_init(w2vmodel, [5,10,15])
pairs,labels=create_pairs(w2vmodel,breaks, nbreaks=[5,10,15])


fnm1='/home/ahmad/duplicate-detection/eventregistrydata/cmfpairs1-b.5.10.15.dat'
mat1=[pairs[:1000],labels[:1000]]
mat1.dump(fnm1)
import cPickle
cPickle.dump( mat1, open(fnm1, "wb"))
'''

fnm='/home/ahmad/duplicate-detection/eventregistrydata/cmfpairs1-b.5.10.15.dat'
mat = np.load(fnm)
#pairs=mat[0]
#labels=mat[1]



pairs,labels=mat[0],mat[1]
ntraining=2*len(pairs)/3
tr_pairs=pairs[:ntraining]
tr_y=labels[:ntraining]
te_pairs=pairs[ntraining:]
te_y=labels[ntraining:]
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
'''
#input_dim = 784
input_dim_r = 40
input_dim_c = 27
epochs = 300
'''
# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(x_test, digit_indices)
'''
# network definition
#base_network = create_base_network(input_dim)
# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
#processed_a = base_network(input_a)
#processed_b = base_network(input_b)

#distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

input_a = Input(shape=(input_dim_r,input_dim_c))
input_b = Input(shape=(input_dim_r,input_dim_c))#0

L1_distance = lambda x: K.abs(x[0]-x[1])#1
both = merge([input_a,input_b], mode = L1_distance, output_shape=lambda x: x[0]) #2
flt = Flatten()(both) #3

'''
distance = Lambda(L1_distance, output_shape=lambda x: x[0])([input_a, input_b]) #1
both = Model([input_a, input_b], output=distance) #2
flt = Flatten()(both.output) #3
'''

'''
maxminbin=lambda x: K.stack([K.max(x,axis=1),K.mean(x,axis=1),K.min(x,axis=1)], axis=1)
maxminbinl = Lambda(maxminbin, output_shape=lambda x: [x[0],1])(both.output)

maxbin=lambda x: K.max(x,axis=1)
maxlambda = Lambda(maxbin, output_shape=lambda x: [x[0],1])(both)
maxbinl = Model(input=both.output, output=maxlambda) #
flt = Flatten()(maxbinl.output) #3
#hidden1 = Dense(32,activation='relu')(maxbinl)
'''

hidden1 = Dense(32,activation='relu')(flt) #4
hidden2 = Dense(16,activation='relu')(hidden1) #5
prediction = Dense(1,activation='sigmoid')(hidden2)#6

'''
intermediate_layer_model = Model(inputs=siamese_net.input, outputs=siamese_net.get_layer(index=6).output)
intermediate_output = intermediate_layer_model.predict([tr_pairs[:1, 0], tr_pairs[:1, 1]])
intermediate_output[0][0][0]
abs(tr_pairs[0, 0][0]- tr_pairs[0, 1][0])

seq = Sequential()
seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
'''

siamese_net = Model(input=[input_a,input_b],output=prediction)
optimizer = Adam(0.00006)
#siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
siamese_net.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
siamese_net.count_params()
siamese_net.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=128, epochs=epochs, validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

'''
siamese_net.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test, y_test))
#model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=128, epochs=epochs, validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
'''
# compute final accuracy on training and test sets
pred = siamese_net.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = siamese_net.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


TP=FP=FN=TN=0
for p,y in zip(pred1,labels):
    if p >0.5 and y == 1:
        TP+=1
    elif p > 0.5 and y == 0:
        FP+=1
    elif p <0.5 and y == 0:
        TN+=1
    elif p <0.5 and y == 1:
        FN+=1

print(TP,TN,FP,FN)
print(100.0*TP/(TP+FN),100.0*TN/(TN+FP),100.0*FP/(TN+FP),100.0*FN/(TP+FN))
>>> print(TP,TN,FP,FN)
8090 26394 7556 25861
>>> print(100.0*TP/(TP+FN),100.0*TN/(TN+FP),100.0*FP/(TN+FP),100.0*FN/(TP+FN))
23.828458661 77.7437407953 22.2562592047 76.171541339

#svm

TP=FP=FN=TN=0
for p,y in zip(pred1,labels):
    if p ==1 and y == 1:
        TP+=1
    elif p ==1 and y == 0:
        FP+=1
    elif p ==0 and y == 0:
        TN+=1
    elif p ==0 and y == 1:
        FN+=1
>>> print(TP,TN,FP,FN)
6554 24668 9282 27397
>>> print(100.0*TP/(TP+FN),100.0*TN/(TN+FP),100.0*FP/(TN+FP),100.0*FN/(TP+FN))
19.3042914789 72.6597938144 27.3402061856 80.6957085211

sum_tr=[sum(vec) for vec in tr_pairs1]

TP=FP=FN=TN=0
for p,y in zip(sum_tr,labels):
    if p <h and y == 1:
        TP+=1
    elif p < h and y == 0:
        FP+=1
    elif p >=h and y == 0:
        TN+=1
    elif p >=h and y == 1:
        FN+=1


diff=[abs(x[0]-x[1]) for x in docpairs]
h=np.percentile(diff,50)
TP=FP=FN=TN=0
for p,y in zip(sum_tr,labels):
    if p <h and y == 1:
        TP+=1
    elif p < h and y == 0:
        FP+=1
    elif p >=h and y == 0:
        TN+=1
    elif p >=h and y == 1:
        FN+=1


print(TP,TN,FP,FN)
print(100.0*TP/(TP+FN),100.0*TN/(TN+FP),100.0*FP/(TN+FP),100.0*FN/(TP+FN))


fnm='/home/ahmad/duplicate-detection/eventregistrydata/cmfpairs-b.5.10.15.dat'
mat = np.load(fnm)
pairs,labels=mat[0],mat[1]
ntraining=2*len(pairs)/3
tr_pairs1=L1_dataflt[:ntraining]
tr_y=labels[:ntraining]
te_pairs1=L1_dataflt[ntraining:]
te_y=labels[ntraining:]
siamese_net.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=128, epochs=epochs, validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

#########################
from sklearn.neighbors import NearestNeighbors
#docpairs,duplabels

print len(w2vpairs),len(labels),len(wordspairs),len(langpairs)

Minpts=5
#pred1=[]
#pred=[]
#truelabel=[]

#preddbscancl1=[]
#preddbscan=[]
Allisdup_labels=[]

Allnumclusters=[]
Allpureclustersratio=[]
Allclustersdist=[]
#Allcosdist=[]
#AllNEdist=[]


posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
allposfiles = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
allnegfiles = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]
to=min(len(allposfiles),len(allnegfiles))
for frm in range(0,to-50,50):
  w2vpairs,labels,wordspairs,langpairs=create_w2v_pairs(unifiedw2vmodel,frm,frm+50)
  print "processing ",docpairs.shape, " pairs"
  numclusters,pureclustersratio,clustersdist=clustering_purity(w2vpairs,dbscan_eps=0.5, dbscan_minPts=5)
  Allclustersdist.extend(clustersdist)
  Allnumclusters.extend(numclusters)
  Allpureclustersratio.extend(pureclustersratio)
  Allisdup_labels.extend(labels)

  for pridx in range(len(w2vpairs1)):
    if w2vpairs[pridx][0].size < 1 or w2vpairs[pridx][1].size < 1:
      continue
    
    numclusters,pureclustersratio,clustersdist=clustering_purity(w2vpairs,dbscan_eps=0.5, dbscan_minPts=5)
    
    Allclustersdist.extend(clustersdist)
    Allnumclusters.extend(numclusters)
    Allpureclustersratio.extend(pureclustersratio)
    Allisdup_labels.extend(labels[pridx])
    
    preddbscancl.append([n_pure_cl,n_noise_cl])
    preddbscan.append(1.0*n_pure_cl/(n_pure_cl+n_noise_cl))
    truelabeldbscan1.append(labels[pridx])

    
    '''
    nbrs = NearestNeighbors(n_neighbors=Minpts, algorithm='ball_tree').fit(X)
    nnn=nbrs.kneighbors_graph(X).toarray()
    npure_nn=0
    nnoisy_nn=0
    for x in range(nnn.shape[0]):
        nnidx=np.where(nnn[x])[0]
        noiseflag=False
        for idx in nnidx:
           if Y[x] != Y[idx]:
               noiseflag=True
               break
        
        if noiseflag:
            nnoisy_nn+=1
        else:
            npure_nn+=1
    
    
    pred1.append([npure_nn,nnoisy_nn])
    pred.append(1.0*npure_nn/(npure_nn+nnoisy_nn))
    truelabel.append(duplabels[pridx])
    '''
for xidx,x in enumerate(docpairs[i][1]):
  if x == w2vmodel[langpairs[i][0]]["la"]:
    print xidx
    break

a=[_x for _x,x in enumerate(wordspairs[i][1]) if x=="shakespeare"]
[docpairs[i][0][_a] for _a in a]
[docpairs[i][1][_a] for _a in a]
b=np.where(w2vmodel[langpairs[i][0]]["shakespeare"][0]==docpairs[i][0])
w2vmodel[langpairs[i][0]]["shakespeare"]
[True for _a in a if _a in b else False] 
for i,p in enumerate(preddbscan1):
  if p[1]>p[0]:
    preddbscan[i]=1-(1.0*p[0]/p[1])
  elif p[0]>p[1]:
    preddbscan[i]=(1.0*p[1]/p[0])-1

for i,p in enumerate(preddbscan1):
  if p[1]>p[0]:
    preddbscan[i]=1.0*p[0]/p[1]
  else:
    preddbscan[i]=-1

for i,p in enumerate(preddbscan1):
   preddbscan[i]=1.0*p[0]/(p[0]+p[1])


pos=[]
neg=[]
for i,z in enumerate(zip(preddbscan1,truelabeldbscan1)):
  #if preddbscan1[i][0]>preddbscan1[i][1]:
  #  continue
  p,l=z
  if l==1:
    pos.append(p)
  elif l==0:
    neg.append(p)

for i,z in enumerate(zip(preddbscan,truelabeldbscan)):
  p,l=z
  if l==1 and p>0.9:
    print i
    break


#plt.savefig("dbscanhist.pdf")
plt.clf()
plt.close()
plt.cla()
plt.hist(neg, weights=np.zeros_like(neg) + 1. / len(neg), color="blue")
plt.savefig("negdbscanhistU.pdf")

plt.clf()
plt.close()
plt.cla()
plt.hist(pos, weights=np.zeros_like(pos) + 1. / len(pos), color="red")
plt.savefig("posdbscanhistU.pdf")

#print(npure_nn,nnoisy_nn)
h=np.mean(preddbscan)
#h=np.percentile(preddbscan,50)
h=0.9
TP=FP=FN=TN=0
FNs=[]
FPs=[]
for i,z in enumerate(zip(preddbscan,truelabeldbscan)):
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

print(TP,TN,FP,FN)
print(100.0*(TP+TN)/(TP+TN+FP+FN),100.0*TP/(TP+FN),100.0*TN/(TN+FP),100.0*FP/(TN+FP),100.0*FN/(TP+FN))
##
a=pred[-len(docpairs):]
b=pred1[-len(docpairs):]
c=truelabel[-len(docpairs):]
##
h=0.25
#h=np.mean(a)
#h=np.percentile(a,15)
TP=FP=FN=TN=0
FNs=[]
FPs=[]
TPs=[]
TNs=[]
for i,z in enumerate(zip(a,c)):
  if b[i][0]>b[i][1]:
    #print i
    continue
  p,l=z
  if p<h and l==1:
    TP+=1
    TPs.append(i)
  elif p<h and l==0:
    FP+=1
    FPs.append(i)
  elif p>h and l==0:
    TN+=1
    TNs.append(i)
  elif p>h and l==1:
    FN+=1
    FNs.append(i)

print(TP,TN,FP,FN)
print(100.0*(TP+TN)/(TP+TN+FP+FN),100.0*TP/(TP+FN),100.0*TN/(TN+FP),100.0*FP/(TN+FP),100.0*FN/(TP+FN))


import matplotlib.pyplot as plt
from sklearn import manifold
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
i=FNs[0]
X = tsne.fit_transform(np.vstack((docpairs[i][0],docpairs[i][1])))
Y=[1]*docpairs[i][0].shape[0]+[2]*docpairs[i][1].shape[0]
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

plt.clf()
plt.close()
plt.cla()
plt.scatter(X[:,0],X[:,1],c=db.labels_+1)
plt.title("DBscan clusters (FN)")
plt.savefig("i"+str(i)+"dbscan.pdf")
#plt.savefig("FP.pdf")

duplabels[FNs[0]]
b[FPs[0]]
langpairs,lbl=create_lang_pairs(w2vmodel,frm,frm+10)
FNlang=dict()
for ii in FNs:
  strstr=str(langpairs[ii])
  if strstr not in FNlang.keys():
    FNlang[strstr]=0
  
  FNlang[strstr]+=1

print FNlang
#FPs: {"['es', 'es']": 98}
#FNs: {"['de', 'en']": 349, "['en', 'es']": 90}
#
alllangpairs
lang=dict()
for ii in range(len(langpairs)):
  
  strstr=str(langpairs[ii])
  if strstr not in lang.keys():
    lang[strstr]=0
  
  lang[strstr]+=1

print lang
#{"['en', 'de']": 368, "['en', 'en']": 612, "['es', 'en']": 48, "['es', 'es']": 84, "['en', 'es']": 389, "['de', 'en']": 601}
#

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=1, min_samples=5).fit(np.vstack((docpairs[i][0],docpairs[i][1])))
np.unique(db.labels_)
import matplotlib.pyplot as plt
plt.clf()
plt.close()
plt.cla()
plt.scatter(X[:,0],X[:,1],c=db.labels_+2)
ll=np.clip(db.labels_,-1,10)
plt.scatter(X[:,0],X[:,1],c=list(ll+1))
plt.scatter(X[:,0],X[:,1],c=2)
for _x,_l ,_y in zip(X, db.labels_, Y):
  plt.scatter(_x[0], _x[1], marker=_y, c=_l)

plt.xlim(min(X[:,0])-50, max(X[:,0])+50)
plt.ylim(min(X[:,1])-50, max(X[:,1])+50)
plt.savefig("FNdbscan.pdf")

ll=np.unique(db.labels_)
n_pure_cl=0
n_noise_cl=0
for _ll in ll:
  idx=np.where(db.labels_==_ll)[0]
  if len(set([Y[_idx] for _idx in idx]))>1:
    n_noise_cl+=1
  else:
    n_pure_cl+=1

print(n_pure_cl,n_noise_cl)