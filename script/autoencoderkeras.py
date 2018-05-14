from leven import levenshtein
import cPickle
#import gzip
import os
import sys
import time
import scipy.io

from keras.models import Model,Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape,Activation
from keras.optimizers import Adam,SGD

#from keras.regularizers import activity_l1
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from keras import losses
from keras.layers.advanced_activations import LeakyReLU, PReLU
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from nltk import StanfordNERTagger
from fasttext import FastVector
import goslate

def loadfasttextmodel(filename):
  filename='/home/ahmad/fastText_multilingual/'
  w2v=dict()
  #['en','es','zh','hr','de','fa','ar','fr']['es','en','de']
  for lng in ['en','es']:
    w2v[lng] = FastVector(vector_file=filename+'wiki.'+lng+'.vec')
    #w2v[lng].apply_transform(filename+'alignment_matrices/'+lng+'.txt')
  
  return w2v



def loadtransfasttextmodel(filename):
  filename='/home/ahmad/fastText_multilingual/'
  w2v=dict()
  #['en','es','zh','hr','de','fa','ar','fr']['es','en','de']
  for lng in ['en','es']:
    w2v[lng] = FastVector(vector_file=filename+'wiki.'+lng+'.vec')
    w2v[lng].apply_transform(filename+'alignment_matrices/'+lng+'.txt')
  
  return w2v

enwords=[]
lng = 'en'
for word in embeddingsmodel0[lng].id2word:
      if len(word)>1:
        enwords.append(word)

with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/FastTextenwords.txt', 'wb') as myfile:
      myfile.write("\n".join(enwords))

print len(enwords)

from google.cloud import translate
from googleapiclient.discovery import build
service = build('translate', 'v2',developerKey='AIzaSyCqpf3hXzheoI9ttfw9JWhMRHtYt5Z72X4')
eswordsTrans=[]

n=len(eswordsTrans)
for idx,word in enumerate(enwords[:5000]):
      if idx < n:
        continue
      try:
        translation=service.translations().list(source='en',target='es',q=[word],format='text').execute()
        eswordsTrans.append(word + " : " + translation['translations'][0]['translatedText'].encode('utf-8'))
      except:
        eswordsTrans.append('')
        pass


with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/FastTextenwordsTranslated_1.txt', 'wb') as myfile:
      myfile.write("\n".join(eswordsTrans))

print sum([True for w in eswordsTrans if len(w)==0]),' out of ',len(eswordsTrans)

for word in eswordsTrans:
  worden,wordes=word.split(":")
  worden=worden.strip()
  wordes=wordes.strip()
  

if __name__ == '__main__':
    
    if len(sys.argv) < 3:
    	print 'Usage: python pretrain_da.py datasetl datasetr learningRate'
    	print 'Example: python pretrain_da.py gauss basic 0.1'
    	sys.exit()
    
    
    #python autoencoderkeras.py /home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNEleftAttro2enes.txt /home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNErightAttro2enes.txt 0.1
    filepathl= str(sys.argv[1]) 
    filepathr = str(sys.argv[2])
    lr = float(sys.argv[3])
    embeddingsmodel0=loadfasttextmodel('Path To Vectors')
    #embeddingsmodel=loadtransfasttextmodel('Path To Vectors')
    vecten=[]
    lng = 'en'
    for word in embeddingsmodel0[lng].id2word:
          vecten.append(embeddingsmodel0[lng][word])
    
    #.reshape(-1,300)[0]
    vectes=[]
    lng = 'es'
    for word in embeddingsmodel0[lng].id2word:
          vectes.append(embeddingsmodel0[lng][word])
    
    vecten=np.asarray(vecten)
    vectes=np.asarray(vectes)
    
    #stanford_ner_path = '/home/ahmad/nltk_data/stanford/stanford-ner.jar'
    #os.environ['CLASSPATH'] = stanford_ner_path
    #stanford_classifier = "/home/ahmad/nltk_data/stanford/es/edu/stanford/nlp/models/ner/spanish.ancora.distsim.s512.crf.ser.gz"
    #stes = StanfordNERTagger(stanford_classifier)
    #stanford_classifier = '/home/ahmad/nltk_data/stanford/english.all.3class.distsim.crf.ser.gz'
    #sten = StanfordNERTagger(stanford_classifier)
    #stanford_classifier = "/home/ahmad/nltk_data/stanford/de/edu/stanford/nlp/models/ner/german.conll.hgc_175m_600.crf.ser.gz"
    #stde = StanfordNERTagger(stanford_classifier)
    
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
    
    
    eswordslist=[]
    for wordidx in range(embeddingsmodel0['es'].n_words):
      if wordidx%10000==0:
        print wordidx
      
      try:
        word1=embeddingsmodel0['es'].id2word[wordidx]
        eswordslist.append(word1)
      except:
          pass
    
    '''
    gs = goslate.Goslate()
    gs.get_languages()['en']
    enwordsl = gs.translate(eswordslist[:5],'en')
    wrd=enwordsl.next()
    y=enwordsl
    enwordslist=list(enwordsl)
    if len(enwordslist)>0:
      with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/FastTextenwordsTranslated.txt', 'wb') as myfile:
        myfile.write("\n".join(enwordslist))
    
    with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/FastTexteswords.txt', 'wb') as myfile:
      myfile.write("\n".join(eswordslist))
    
    enwordslist=[]
    for idx in range(0,len(eswordslist)-99,100):
      wlist = eswordslist[idx:idx+100]
      gs = goslate.Goslate()
      enwordslist = gs.translate(eswordslist)
      translation = list(translation_iter)
    
      translations = translator.translate(wlist, dest='en')
      for translation in translations:
        enwordslist.append(translation.text)
      
      break
    '''
    
    allNE=[]
    data_en=[]
    data_es=[]
    wordslist=[]
    for wordidx in range(embeddingsmodel0['es'].n_words):
      if wordidx%10000==0:
        print wordidx
      
      try:
        word1=embeddingsmodel0['es'].id2word[wordidx]
        
        data_en.append(embeddingsmodel0['en'][word1])
        data_es.append(embeddingsmodel0['es'][word1])
        wordslist.append(word1)
      except:
          pass
    
    print len(data_en),len(data_es),len(wordslist)
    
    '''
    # Imports the Google Cloud client library
    from google.cloud import translate
    translate_client = translate.Client()
    text = u'Hello, world!'
    target = 'ru'
    
    # Translates some text into Russian
    translation = translate_client.translate( text, target_language=target)
    print(u'Text: {}'.format(text))
    print(u'Translation: {}'.format(translation['translatedText']))
    
    from googletrans import Translator
    translator = Translator()
    translator.translate('?????.').text
    
    translations = translator.translate([], dest='en', src='es')
    for translation in translations:
      print(translation.origin, ' -> ', translation.text)
    
    import goslate
    gs = goslate.Goslate()
    print(gs.translate('hello world', 'de'))
    
    
    gs = goslate.Goslate()
    translation_iter = gs.translate(open(big_file, 'r').read() for big_file in big_files)
    translation = list(translation_iter)
    '''
    
    
    '''
    allNE=[]
    for _leftNE, _rightNE in zip(leftNE,rightNE):
      word1 = _leftNE[0]
      word2 = _rightNE[0]
      
      if word1+word2 in allNE:
        continue
      else:
        allNE.append(word1+word2)
      
      if word1!= word2:
        continue
      
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
          
          if embeddingsmodel0['es'].__contains__(word2) and embeddingsmodel0['en'].__contains__(word1):
            wordslist.append(word1)
            data_es.append(embeddingsmodel0['es'][word2])
            data_en.append(embeddingsmodel0['en'][word1])
        except:
          pass
    
    '''
    data_es=np.asarray(data_es)
    data_en=np.asarray(data_en)
    print len(allNE),data_es.shape,data_en.shape
    
    entree = spatial.cKDTree(vecten)
    #dden, iien = entree.query(data_en, k=3, n_jobs=14)
    #dd, ii = entree.query(data_l[0,], k=5, n_jobs=14)
    #embeddingsmodel0['en'].id2word[ii[2]]
    #word='universidad'
    #dd, ii = estree.query(embeddingsmodel0['es'][word], k=5, n_jobs=14)
    #embeddingsmodel0['es'].id2word[ii[0]]
    
    estree = spatial.cKDTree(vectes)
    
    dden, iien = entree.query(data_en, k=2, n_jobs=14)
    ddes, iies = estree.query(data_es, k=2, n_jobs=14)
    
    print "1-NN..."
    data_en=list(data_en)
    for ii in iien:
      for i in ii[1:]:
        data_en.append(vecten[i])
    
    data_es=list(data_es)
    for ii in iies:
      for i in ii[1:]:
        data_es.append(vectes[i])
    
    data_en=np.asarray(data_en)
    data_es=np.asarray(data_es)
    
    print len(allNE),data_en.shape,data_es.shape
    
    dim=data_en.shape[1]
    model = Sequential()
    model.add(Dense(9*dim, activation='linear', activity_regularizer=regularizers.l1(0.00000000001),input_shape=(dim,)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(3*dim/4, activation='linear', activity_regularizer=regularizers.l1(0.00000000001)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(2*dim/3, activation='linear', activity_regularizer=regularizers.l1(0.00000000001), name='encoderlayer'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(3*dim/4, activation='linear', activity_regularizer=regularizers.l1(0.00000000001)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(9*dim/10, activation='linear', activity_regularizer=regularizers.l1(0.00000000001)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(dim))
    
    #layer_name = 'encoderlayer'
    #intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    
    model.compile(optimizer='Adam', loss=losses.cosine_proximity)
    model.fit(data_es, data_en, batch_size=64, epochs=10)
    #model.fit(data_es, data_en, verbose=1, batch_size=64, epochs=10)
    for word in [wordslist[0],wordslist[-1]]:
      print cosine_similarity(model.predict(embeddingsmodel0['es'][word].reshape(1,-1)),embeddingsmodel0['en'][word].reshape(1,-1))#,cosine_similarity(embeddingsmodel['es'][word].reshape(1,-1),embeddingsmodel['en'][word].reshape(1,-1))
    
    print "Transforming model..."
    transformedmodel=dict()
    lng='es'
    transformedmodel[lng]=dict()
    for word in embeddingsmodel0[lng].id2word:
      transformedmodel[lng][word] = model.predict(embeddingsmodel0[lng][word].reshape(1,-1))[0]
    
    lng='en'
    transformedmodel[lng]=dict()
    for word in embeddingsmodel0[lng].id2word:
      transformedmodel[lng][word] = embeddingsmodel0[lng][word]
    
    print "Saving model..."
    cPickle.dump(transformedmodel, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/FastTextNNAdamcosLeakyReLU10epcs3Layers.p', 'wb'))
    #cPickle.dump(transformedmodel, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/FastTextAttro2enesMINMAXAdamcoshtanh10epcs3Layers2ways.p', 'wb'))




'''
    
    #data_l = data_l / allvectnorm#np.linalg.norm(data_l)
    #data_r = data_r / allvectnorm#np.linalg.norm(data_r)
    
    #data_l = (data_l-vectmean)/vectstd
    #data_r = (data_r-vectmean)/vectstd
    #data_l = (data_l-vectmin)/(vectmax-vectmin)
    #data_r = (data_r-vectmin)/(vectmax-vectmin)
    
    #noise_factor = 1.0
    #data_l = data_l + noise_factor * np.random.normal(loc=0.0, scale=0.5, size=data_l.shape) 
    #data_l = np.clip(data_l, 0., 1.)
    
    inputs = Input(shape=(data_l.shape[1],))
    #hencode = Dense(3*data_l.shape[1]/4, activation='relu')(inputs)
    #hencode2 = Dense(2*data_l.shape[1]/3, activation='tanh')(hencode) #activity_l1(1e-5)
    encoded = Dense(2*data_l.shape[1]/3, activation='relu', activity_regularizer=regularizers.l1(0.01))(inputs) #activity_l1(1e-5)
    #hdecode2 = Dense(2*data_l.shape[1]/3, activation='tanh')(encoded) #activity_l1(1e-5)relu
    #hdecode = Dense(3*data_l.shape[1]/4, activation='relu')(encoded)
    outputs = Dense(data_l.shape[1])(encoded)
    model = Model(inputs=inputs, outputs=outputs)
    
    encoder = Model(inputs=inputs, outputs=encoded)
    #encoded_input = Input(shape=(data_l.shape[1]/2,))
    #decoder_layer = model.layers[len(model.layers)/2]
    #decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=1.9)
    model.compile(optimizer='sgd', loss='cosine_proximity')#mse#cosine_proximity#sgd#logcosh#losses.kullback_leibler_divergence#binary_crossentropy
    model.fit(data_l, data_r, batch_size=64, epochs=20)
    
    
    
    transformedmodel=dict()
    lng='en'
    transformedmodel[lng]=dict()
    for word in embeddingsmodel[lng].id2word:
          #transformedmodel[lng][word] = encoder.predict(embeddingsmodel[lng][word].reshape(1,-1))
          transformedmodel[lng][word] = model.predict(embeddingsmodel[lng][word].reshape(-1,data_r.shape[1]))
          #transformedmodel[lng][word] = model.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))
          #transformedmodel[lng][word] = encoder.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))
    
    lng='es'
    transformedmodel[lng]=dict()
    for word in embeddingsmodel[lng].id2word:
          #transformedmodel[lng][word] = (((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin)
          transformedmodel[lng][word] = (embeddingsmodel[lng][word].reshape(-1,data_r.shape[1]))
          #transformedmodel[lng][word] = encoder.predict(embeddingsmodel[lng][word].reshape(1,-1))#encoder.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))
          #transformedmodel[lng][word] = model.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))
          #transformedmodel[lng][word] = encoder.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))
    
    
    inputs = Input(shape=(data_r.shape[1],))
    encoded = Dense(2*data_l.shape[1]/3, activation='tanh', activity_regularizer=regularizers.l1(1e-5))(inputs) #activity_l1(1e-5)
    outputs = Dense(data_r.shape[1])(encoded)
    model = Model(input=inputs, output=outputs)
    encoder = Model(input=inputs, output=inputs)
    model.compile(optimizer='Adam', loss=losses.cosine_proximity)#mse#cosine_proximity#sgd#logcoshlosses.kullback_leibler_divergence
    model.fit(data_r, data_l, batch_size=64, epochs=10)
    
    lng='es'
    transformedmodel[lng]=dict()
    for word in embeddingsmodel[lng].id2word:
          transformedmodel[lng][word] = encoder.predict(embeddingsmodel[lng][word].reshape(-1,data_r.shape[1]))
    
    
    transformedmodel=dict()
    for lng in embeddingsmodel.keys():
      transformedmodel[lng]=dict()
      if lng=='es':
        for word in embeddingsmodel[lng].id2word:
          transformedmodel[lng][word]=embeddingsmodel[lng][word]
      else:
        for word in embeddingsmodel[lng].id2word:
          encoded_data = encoder.predict(embeddingsmodel[lng][word].reshape(-1,data_l.shape[1]))
          transformedmodel[lng][word]=decoder.predict(encoded_data)
    
    cPickle.dump(transformedmodel, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/FastTextAttro2enesMINMAXAdamcoshtanh10epcs3Layers2ways.p', 'wb'))



data_l=[]
data_r=[]
wordslist=[]
allNE=[]
for _leftNE, _rightNE in zip(leftNE,rightNE):
  word1 = _leftNE[0]
  word2 = _rightNE[0]
  if word1+word2 in allNE:
    continue
  else:
    allNE.append(word1+word2)
  
  if word1 != word2:
    continue
  
  if 1.0 * levenshtein(word1, word2)/(len(word1)+len(word2))>0.20:
    continue
  
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
        wordslist.append(word1+' '+word2)
        data_r.append(embeddingsmodel['es'][word2])
        data_l.append(embeddingsmodel['en'][word1])
        #data_l.append(embeddingsmodel['es'][word2])
        #data_r.append(embeddingsmodel['en'][word1])
    except:
      pass

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

#del embeddingsmodel

data_l=np.asarray(data_l)
data_r=np.asarray(data_r)
print len(allNE),data_l.shape,data_r.shape

data_l0=np.asarray(data_l0)
data_r0=np.asarray(data_r0)
print len(allNE),data_l0.shape,data_r0.shape

#data_l = data_l / allvectnorm#np.linalg.norm(data_l)
#data_r = data_r / allvectnorm#np.linalg.norm(data_r)

#data_l = (data_l-vectmean)/vectstd
#data_r = (data_r-vectmean)/vectstd
#data_l = (data_l-vectmin)/(vectmax-vectmin)
#data_r = (data_r-vectmin)/(vectmax-vectmin)

#noise_factor = 1.0
#data_l = data_l + noise_factor * np.random.normal(loc=0.0, scale=0.5, size=data_l.shape) 
#data_l = np.clip(data_l, 0., 1.)

linear
tanh
#model.add(LeakyReLU(alpha=0.3))

model = Sequential()
model.add(Dense(9*data_l.shape[1]/10, activation='linear', activity_regularizer=regularizers.l1(0.00000000001),input_shape=(data_l.shape[1],)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(3*data_l.shape[1]/4, activation='linear', activity_regularizer=regularizers.l1(0.00000000001)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(2*data_l.shape[1]/3, activation='linear', activity_regularizer=regularizers.l1(0.00000000001), name='my_layer'))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(3*data_l.shape[1]/4, activation='linear', activity_regularizer=regularizers.l1(0.00000000001)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(9*data_l.shape[1]/10, activation='linear', activity_regularizer=regularizers.l1(0.00000000001)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(data_l.shape[1]))

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

model.compile(optimizer='Adam', loss=losses.cosine_proximity)
model.fit(data_l0, data_r0, verbose=0, batch_size=64, epochs=2000)
for word in [wordslist[0].split(" ")[0],wordslist[105].split(" ")[0]]:
  print cosine_similarity(model.predict(embeddingsmodel0['en'][word].reshape(-1,data_r.shape[1])),embeddingsmodel0['es'][word].reshape(1,-1)),cosine_similarity(embeddingsmodel['en'][word].reshape(-1,data_r.shape[1]),embeddingsmodel['es'][word].reshape(1,-1))

wordslist1=[wr.split(" ")[0] for wr in wordslist]]

word1 = 'document'
word2 = 'documento'

word1 = wordslist[0].split(" ")[0]
word2 = wordslist[1].split(" ")[0]
print cosine_similarity(model.predict(embeddingsmodel0['en'][word1].reshape(-1,data_r.shape[1])),embeddingsmodel0['es'][word2].reshape(1,-1))
print cosine_similarity(embeddingsmodel['en'][word1].reshape(-1,data_r.shape[1]),embeddingsmodel['es'][word2].reshape(1,-1))
print cosine_similarity(model.predict(embeddingsmodel0['en'][word1].reshape(-1,data_r.shape[1])),embeddingsmodel0['es'][word1].reshape(1,-1))
print cosine_similarity(embeddingsmodel['en'][word1].reshape(-1,data_r.shape[1]),embeddingsmodel['es'][word1].reshape(1,-1))

pred=model.predict(embeddingsmodel['en']['barack'].reshape(-1,data_r.shape[1]))
preden = intermediate_layer_model.predict(embeddingsmodel['en']['barack'].reshape(-1,data_r.shape[1]))
predes = intermediate_layer_model.predict(embeddingsmodel['es']['barack'].reshape(-1,data_r.shape[1]))
print cosine_similarity(pred,embeddingsmodel['es']['barack'].reshape(1,-1))
print cosine_similarity(preden,predes)
print cosine_similarity(embeddingsmodel['en']['barack'].reshape(1,-1),embeddingsmodel['es']['barack'].reshape(1,-1))


inputs = Input(shape=(data_l.shape[1],))
#hencode = Dense(3*data_l.shape[1]/4, activation='relu')(inputs)
#hencode2 = Dense(2*data_l.shape[1]/3, activation='tanh')(hencode) #activity_l1(1e-5)
activationlayer=LeakyReLU(alpha=0.3)
encoded = Dense(2*data_l.shape[1]/3, activation='linear', activity_regularizer=regularizers.l1(0.01))(inputs) #activity_l1(1e-5)
model.add(Dense(512, 512, activation='linear')) # Add any layer, with the default of an identity/linear squashing function (no squashing)
model.add(LeakyReLU(alpha=.001))

#hdecode2 = Dense(2*data_l.shape[1]/3, activation='tanh')(encoded) #activity_l1(1e-5)relu
#hdecode = Dense(3*data_l.shape[1]/4, activation='relu')(encoded)
outputs = Dense(data_l.shape[1])(encoded)
model = Model(inputs=inputs, outputs=outputs)

encoder_layer = model.layers[1]
encoder = Model(inputs=inputs, outputs=encoder_layer(inputs))
#encoded_input = Input(shape=(data_l.shape[1]/2,))
#decoder_layer = model.layers[len(model.layers)/2]
#decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=1.9)
#mse#cosine_proximity#sgd#logcosh#losses.kullback_leibler_divergence#binary_crossentropy
model.compile(optimizer='Adam', loss=losses.cosine_proximity)
model.fit(data_l[:100], data_r[:100], verbose=0, batch_size=64, epochs=2000)
pred=model.predict(embeddingsmodel['en']['barack'].reshape(-1,data_r.shape[1]))
preden=encoder.predict(embeddingsmodel['en']['barack'].reshape(-1,data_r.shape[1]))
predes=encoder.predict(embeddingsmodel['es']['barack'].reshape(-1,data_r.shape[1]))
print cosine_similarity(pred,embeddingsmodel['es']['barack'].reshape(1,-1))
print cosine_similarity(preden,predes)
print cosine_similarity(embeddingsmodel['en']['barack'].reshape(1,-1),embeddingsmodel['es']['barack'].reshape(1,-1))

transformedmodel=dict()
lng='en'
transformedmodel[lng]=dict()
for word in embeddingsmodel[lng].id2word:
      #transformedmodel[lng][word] = encoder.predict(embeddingsmodel[lng][word].reshape(1,-1))
      transformedmodel[lng][word] = model.predict(embeddingsmodel0[lng][word].reshape(-1,data_r.shape[1]))[0]
      #transformedmodel[lng][word] = model.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))
      #transformedmodel[lng][word] = encoder.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))

lng='es'
transformedmodel[lng]=dict()
for word in embeddingsmodel[lng].id2word:
      #transformedmodel[lng][word] = (((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin)
      transformedmodel[lng][word] = (embeddingsmodel0[lng][word].reshape(-1,data_r.shape[1]))[0]
      #transformedmodel[lng][word] = encoder.predict(embeddingsmodel[lng][word].reshape(1,-1))#encoder.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))
      #transformedmodel[lng][word] = model.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))
      #transformedmodel[lng][word] = encoder.predict((((embeddingsmodel[lng][word].reshape(-1,data_r.shape[1])-vectmean)/vectstd) -vectmin)/(vectmax-vectmin))

'''