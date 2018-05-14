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
        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))
        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            initial_W = np.asarray(np_rng.uniform(low=-4 * np.sqrt(6. / (n_hidden + n_visible)),high=4 * np.sqrt(6. / (n_hidden + n_visible)),size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)
        if not bvis:
            bvis = theano.shared(value=np.zeros(n_visible,dtype=theano.config.floatX),borrow=True)
        if not bhid:
            bhid = theano.shared(value=np.zeros(n_hidden,dtype=theano.config.floatX),name='b',borrow=True)
        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input == None:
            self.x = T.dmatrix(name='input')
            self.y = T.dmatrix(name='input')
        else:
            self.x,self.y = input
        
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
    
    
    
    
    
    def get_cost_updates(self, learning_rate):
        
        '''
        def compile_cos_sim_theano():
          v1 = theano.tensor.vector(dtype=theano.config.floatX)
          v2 = theano.tensor.vector(dtype=theano.config.floatX)
          numerator = theano.tensor.sum(v1*v2)
          denominator = theano.tensor.sqrt(theano.tensor.sum(v1**2)*theano.tensor.sum(v2**2))
          return theano.function([v1, v2], numerator/denominator)
        
        cos_sim_theano_fn = compile_cos_sim_theano()
        '''
        def _squared_magnitude( x):
            return T.sqr(x).sum(axis=-1)
        
        def _magnitude( x):
            return T.sqrt(T.maximum(_squared_magnitude(x), np.finfo(x.dtype).tiny))
        
        def cosine( x, y):
            return T.clip((1 - (x * y).sum(axis=-1) / (_magnitude(x) * _magnitude(y))) / 2, 0, 1)
        
        #print self.y.eval()
        #cost = T.mean(cos_sim_theano_fn(self.x, self.y))
        #cost = T.mean(T.clip((1 - (self.x * self.y).sum(axis=-1) / (T.sqrt(T.maximum(T.sqr(self.x).sum(axis=-1), np.finfo(self.x.dtype).tiny)) * T.sqrt(T.maximum(T.sqr(self.y).sum(axis=-1), np.finfo(self.y.dtype).tiny)))) / 2, 0, 1))
        #cost = T.mean(cosine(self.x, self.y))
        #print cost.eval()
        #tilde_x = self.get_corrupted_input(self.x, noisemodel, noiserate)
        hdn = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(hdn)
        #L = - T.sum(self.y * T.log(z) + (1 - self.y) * T.log(1 - z), axis=1)
        #L = T.sqrt(T.sum(T.pow(T.sub(self.y , z),2)))
        L = T.sum(T.abs_(T.sub(self.y , z)), axis=1)
        #print (L.eval()).shape
        #L = cosine(self.y,z)
        cost = T.mean(L)
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        
        return (cost, updates)

def mytest_dA(lefttr,righttr, learning_rate, training_epochs=300, batch_size=20):
    tunning_epochs = range(1,20,2)+range(20, 301, 20);
    #datasets = load_data(dataset)
    #train_set_x=mytrainingdata
    #train_set_x, train_set_y = datasets[0]
    
    print lefttr.get_value(borrow=True).shape
    print righttr.get_value(borrow=True).shape
    # compute number of minibatches for training, validation and testing
    n_train_batches = lefttr.get_value(borrow=True).shape[0] / batch_size
    # allocate symbolic variables for the data
    index = T.lscalar()# index to a [mini]batch
    x = T.matrix('x')
    y = T.matrix('y')
    
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    #da = dA(np_rng=rng, theano_rng=theano_rng, input=x,n_visible=28*28, n_hidden=1000)
    da = dA(np_rng=rng, theano_rng=theano_rng, input=[x,y], n_visible=lefttr.get_value(borrow=True).shape[1],n_hidden=lefttr.get_value(borrow=True).shape[1]/3)# n_hidden=100)    
    cost, updates = da.get_cost_updates(learning_rate=learning_rate)
    train_da = theano.function([index], cost, updates=updates, givens={x: lefttr[index * batch_size:(index + 1) * batch_size],y:righttr[index * batch_size:(index + 1) * batch_size]})
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



if __name__ == '__main__':
    
    if len(sys.argv) < 3:
    	print 'Usage: python pretrain_da.py datasetl datasetr learningRate'
    	print 'Example: python pretrain_da.py gauss basic 0.1'
    	sys.exit()
    
    #python autoencoder.py /matchingwordsleftAttro2enes.txt /matchingwordsrightAttro2enes.txt 0.1
    #python autoencoder.py /home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNEleftAttro2enes.txt /home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNErightAttro2enes.txt 0.1
    filepathl= str(sys.argv[1]) 
    filepathr = str(sys.argv[2])
    lr = float(sys.argv[3])
    
    ''' 
    log_folder = root_folder + '/logs/da_' + mm + '/layer'+str(input_ll+1)
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)	
    logfile = open(os.path.join(log_folder, dd+',lr='+str(lr)+'.log'), 'w', 0)
    sys.stdout = logfile
    '''
    
    #filepathl='/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNEleftAttro2enes.txt'
    #filepathr='/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNErightAttro2enes.txt'
    leftNE=[]
    with open(filepathl,'r') as myfile:
      leftNE=myfile.readlines()
    
    rightNE=[]
    with open(filepathr,'r') as myfile:
      rightNE=myfile.readlines()
    
    leftNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in leftNE]
    rightNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in rightNE]
    data_l=[]
    data_r=[]
    for _leftNE, _rightNE in zip(leftNE,rightNE):
      word1 = _leftNE[0]
      word2 = _rightNE[0]
      if '' !=word1.strip().lower() and '' !=word2.strip().lower():
        try:
          if type(word1)==type(''):
            word1=word1.strip().lower().decode('utf-8')
          else:
            word1=word1.strip().lower()
          
          if type(word2)==type(''):
            word2=word2.strip().lower().decode('utf-8')
          else:
            word2=word2.strip().lower()
          
          if embeddingsmodel['es'].__contains__(word2) and embeddingsmodel['en'].__contains__(word1):
            data_r.append(embeddingsmodel['es'][word2])
            data_l.append(embeddingsmodel['en'][word1])
        except:
          pass
    
    del embeddingsmodel
    data_l=np.asarray(data_l)
    data_r=np.asarray(data_r)
    
    data_l = data_l / np.linalg.norm(data_l)
    data_r = data_r / np.linalg.norm(data_r)
    #norm2 = normalize(data_l, axis=0).ravel()
    
    #print data_l.shape,data_r.shape
    
    '''
    with open(filepathl, 'r') as myfile:
        content=myfile.readlines()#.replace('\n', '')
    
    def str2float(strng):
      if strng.strip()=="":
        return strng.strip()
      return float(strng.strip())
    
    data_l=[map(str2float,con.strip()[1:-1].split(",")) for con in content if "," in con]
    data_l=np.asarray(data_l)
    #print data_l
    with open(filepathr, 'r') as myfile:
        content=myfile.readlines()#.replace('\n', '')
    
    data_r=[map(str2float,con.strip()[1:-1].split(",")) for con in content if "," in con]
    data_r=np.asarray(data_r)
    '''
    #print data_r
    shared_x = theano.shared(np.asarray(data_l, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(np.asarray(data_r, dtype=theano.config.floatX), borrow=True)
    Weights=mytest_dA(lefttr=shared_x,righttr=shared_y, learning_rate=lr, training_epochs=1000, batch_size=20)
    W=Weights['W']
    b=Weights['b']
    cPickle.dump(Weights, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/WeightsAttro2enesNormL11000epcs1.p', 'wb'))
    
    data_l_temp=[]
    data_r_temp=[]
    for idx,_data_l,_data_r in zip(range(len(data_l)),data_l,data_r):
      if len(_data_l)< W.shape[0] and len(_data_r)< W.shape[0]:
        continue
      
      hx = np.dot(_data_l, W) +b
      xr = (1./(1+np.exp(-hx)))
      data_l_temp.append(xr)
      hx = np.dot(_data_r, W) +b
      xr = (1./(1+np.exp(-hx)))
      data_r_temp.append(xr)
    
    data_l=np.asarray(data_l_temp)
    data_r=np.asarray(data_r_temp)
    shared_x = theano.shared(np.asarray(data_l, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(np.asarray(data_r, dtype=theano.config.floatX), borrow=True)
    Weights=mytest_dA(lefttr=shared_x,righttr=shared_y, learning_rate=lr, training_epochs=1000, batch_size=20)
    W=Weights['W']
    b=Weights['b']
    cPickle.dump(Weights, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/WeightsAttro2enesNormL11000epcs2.p', 'wb'))
    
    data_l_temp=[]
    data_r_temp=[]
    for idx,_data_l,_data_r in zip(range(len(data_l)),data_l,data_r):
      if len(_data_l)< W.shape[0] and len(_data_r)< W.shape[0]:
        continue
      
      hx = np.dot(_data_l, W) +b
      xr = (1./(1+np.exp(-hx)))
      data_l_temp.append(xr)
      hx = np.dot(_data_r, W) +b
      xr = (1./(1+np.exp(-hx)))
      data_r_temp.append(xr)
    
    data_l=np.asarray(data_l_temp)
    data_r=np.asarray(data_r_temp)
    shared_x = theano.shared(np.asarray(data_l, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(np.asarray(data_r, dtype=theano.config.floatX), borrow=True)
    Weights=mytest_dA(lefttr=shared_x,righttr=shared_y, learning_rate=lr, training_epochs=1000, batch_size=20)
    W=Weights['W']
    b=Weights['b']
    cPickle.dump(Weights, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/WeightsAttro2enesNormL11000epcs3.p', 'wb'))



