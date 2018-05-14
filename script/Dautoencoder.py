import cPickle
import gzip
import os
import sys
import time
import scipy.io

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from scipy.io import arff
from cStringIO import StringIO


input_ll=0

class dA(object):
    def __init__(self, np_rng, theano_rng=None, input=None,n_visible=784, n_hidden=1000,W=None, bhid=None, bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))
        if not W:
            initial_W = np.asarray(np_rng.uniform(low=-4 * np.sqrt(6. / (n_hidden + n_visible)),high=4 * np.sqrt(6. / (n_hidden + n_visible)),size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)
        
        if not bvis:
            bvis = theano.shared(value=np.zeros(n_visible, dtype=theano.config.floatX), borrow=True)
        
        if not bhid:
            bhid = theano.shared(value=np.zeros(n_hidden, dtype=theano.config.floatX), name='b', borrow=True)
        
        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        if input == None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        
        self.params = [self.W, self.b, self.b_prime]
    
    def get_corrupted_input(self, input, noisemodel, noiserate):
      if noisemodel == 'dropout':
      	return  self.theano_rng.binomial(size=input.shape, n=1, p=1-noiserate, dtype=theano.config.floatX) * input
      else:
      	return self.theano_rng.normal(size=input.shape, avg=0.0, std=noiserate, dtype=theano.config.floatX) + input
    
    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
    
    def get_reconstructed_input(self, hidden):
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
    
    
    
    
    
    def get_cost_updates(self, noisemodel, noiserate, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, noisemodel, noiserate)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)
        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        
        return (cost, updates)

def mytest_dA(mytrainingdata, learning_rate, noisemodel, noiserange, training_epochs=300, batch_size=20):
    tunning_epochs = range(1,20,2)+range(20, 301, 20);
    print mytrainingdata.get_value(borrow=True).shape
    # compute number of minibatches for training, validation and testing
    n_train_batches = mytrainingdata.get_value(borrow=True).shape[0] / batch_size
    # allocate symbolic variables for the data
    index = T.lscalar()# index to a [mini]batch
    x = T.matrix('x')
    
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    #da = dA(np_rng=rng, theano_rng=theano_rng, input=x,n_visible=28*28, n_hidden=1000)
    da = dA(np_rng=rng, theano_rng=theano_rng, input=x, n_visible=mytrainingdata.get_value(borrow=True).shape[1],n_hidden=mytrainingdata.get_value(borrow=True).shape[1]/3)# n_hidden=100)    
    cost, updates = da.get_cost_updates(noisemodel=noisemodel, noiserate=noiserange,learning_rate=learning_rate)
    train_da = theano.function([index], cost, updates=updates, givens={x: mytrainingdata[index * batch_size:(index + 1) * batch_size]})
    #train_da = theano.function([index], cost, updates=updates, givens=[(lefttr[index * batch_size:(index + 1) * batch_size],righttr[index * batch_size:(index + 1) * batch_size])])
    ############
    # TRAINING #
    ############
    # go through training epochs
    start_time = time.time()
    for epoch in range(1, training_epochs+1):
      c = []
      for batch_index in xrange(n_train_batches):
        l = train_da(batch_index)
        c.append(l)
      
      if epoch%100 == 0:
        print 'Training epoch %d, cost %f' % (epoch, np.mean(c))
    
    training_time = time.time() - start_time
    print 'learning rate %.6f' %(learning_rate) + ' finished in %.2fm' % ((training_time) / 60.)
    return {'W': da.W.get_value(borrow=True), 'b':da.b.get_value(borrow=True), 'b_prime':da.b_prime.get_value(borrow=True), 'cost':c}


def loadw2vmodel(filename):
  w2v=dict()
  #filename='fifty_nine.table5.multiCluster.m_1000+iter_10+window_3+min_count_5+size_40.normalized'
  #filename='/home/ahmad/duplicate-detection/multilingual-embedding/three.table4.multiSkip.iter_10+window_3+min_count_5+size_40.normalized'
  with open(filename, "r") as myfile:
    for line in myfile:
      lineparts=line.strip().split(":")
      #if lineparts[0] not in ['fr','en','es', 'zh', 'hr', 'de']:
      if lineparts[0] not in ['en','es']:
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


if __name__ == '__main__':
    #if len(sys.argv) < 2:
    #	print 'Usage: python pretrain_da.py datasetl learningRate'
    #	#print 'Example: python pretrain_da.py gauss basic 0.1'
    #	sys.exit()
    
    w2vmodel=loadw2vmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
    mydata=[]
    for ky in w2vmodel['en'].keys():
      if len(w2vmodel['en'][ky])>10:
        mydata.append(w2vmodel['en'][ky])
    
    mydata=np.asarray(mydata)
    
    #print mydata.shape
    shared_x = theano.shared(np.asarray(mydata, dtype=theano.config.floatX), borrow=True)
    
    Weights=mytest_dA(mytrainingdata=shared_x, learning_rate=0.01,noisemodel='normal',noiserange=1.0, training_epochs=300, batch_size=20)
    W=Weights['W']
    b=Weights['b']
    cPickle.dump(Weights, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/WeightsAttro2enDAE.p', 'wb'))
    
    print "spanish..."
    mydata=[]
    for ky in w2vmodel['es'].keys():
      if len(w2vmodel['es'][ky])>10:
        mydata.append(w2vmodel['es'][ky])
    
    mydata=np.asarray(mydata)
    
    #print mydata.shape
    shared_x = theano.shared(np.asarray(mydata, dtype=theano.config.floatX), borrow=True)
    
    Weights=mytest_dA(mytrainingdata=shared_x, learning_rate=0.01,noisemodel='normal',noiserange=1.0, training_epochs=300, batch_size=20)
    W=Weights['W']
    b=Weights['b']
    cPickle.dump(Weights, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/WeightsAttro2esDAE.p', 'wb'))