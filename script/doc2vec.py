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
import numpy
from sklearn.cluster import KMeans


def get_doc_list(sentencesfile):
  dataText=[]
  f = open(sentencesfile, 'r')
  for d in f:
    dataText.append(d.decode('utf-8'))
  
  return dataText

def get_doc(sentencesfile):
  doc_list = get_doc_list(sentencesfile)
  tokenizer = RegexpTokenizer(r'\w+')
  #en_stop = get_stop_words('en')
  en_stop = set(stopwords.words('english'))
  p_stemmer = PorterStemmer() 
  taggeddoc = []
  texts = []
  for index,i in enumerate(doc_list):
      # for tagged doc
      wordslist = []
      tagslist = []
      # clean and tokenize document string
      raw = i.lower()
      tokens = tokenizer.tokenize(raw)
      # remove stop words from tokens
      stopped_tokens = [i for i in tokens if not i in en_stop]
      # remove numbers
      number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
      number_tokens = ' '.join(number_tokens).split()
      # stem tokens
      stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
      # remove empty
      length_tokens = [i for i in stemmed_tokens if len(i) > 1]
      # add tokens to list
      texts.append(length_tokens)
      td = TaggedDocument((' '.join(stemmed_tokens)).split(),[str(index)])
      #td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(),str(index))
      taggeddoc.append(td)
  
  return taggeddoc


def sents_to_taggeddoc(doc_list):
  #doc_list = get_doc_list(sentencesfile)
  tokenizer = RegexpTokenizer(r'\w+')
  #en_stop = get_stop_words('en')
  en_stop = set(stopwords.words('english'))
  p_stemmer = PorterStemmer() 
  taggeddoc = []
  texts = []
  for index,i in enumerate(doc_list):
      # for tagged doc
      wordslist = []
      tagslist = []
      # clean and tokenize document string
      raw = i.lower()
      tokens = tokenizer.tokenize(raw)
      # remove stop words from tokens
      stopped_tokens = [i for i in tokens if not i in en_stop]
      # remove numbers
      number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
      number_tokens = ' '.join(number_tokens).split()
      # stem tokens
      stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
      # remove empty
      length_tokens = [i for i in stemmed_tokens if len(i) > 1]
      # add tokens to list
      texts.append(length_tokens)
      #td = TaggedDocument((' '.join(stemmed_tokens)).split(),str(index))
      td = TaggedDocument((' '.join(stemmed_tokens)).split(),[str(index)])
      #td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(),str(index))
      taggeddoc.append(td)
  
  return taggeddoc


def trainmodel(documentsbatch):
  #model = gensim.models.Doc2Vec(documentsbatch, size=100, window=8, min_count=5, workers=4)
  #model = gensim.models.Doc2Vec(documentsbatch, dm = 0, alpha=0.025, size= 20, min_alpha=0.025, min_count=0)
  model = gensim.models.Doc2Vec(documentsbatch, size= 20)
  for epoch in range(200):
      if epoch % 50 == 0:
          print ('Now training epoch %s'%epoch)
      
      model.train(documentsbatch)
      #model.alpha -= 0.002  # decrease the learning rate
      #model.min_alpha = model.alpha  # fix the learning rate, no decay
  
  return model

if __name__ == '__main__':
  sentencesfile='../atrocitiesdata/TextList3.txt'
  dataText=get_doc_list(sentencesfile)
  documents =sents_to_taggeddoc(dataText)
  #model=trainmodel(documents)
  '''
  data=[]
  for i in range(len(model.docvecs)):
    data.append(model.docvecs[i])
  
  data=numpy.asarray(data)
  
  # build the model
  #model = gensim.models.Doc2Vec(documents, dm = 0, alpha=0.025, size= 20, min_alpha=0.025, min_count=0)
  model = gensim.models.Doc2Vec(documents, size=100, window=8, min_count=5, workers=4)
  # start training
  for epoch in range(20):
      if epoch % 5 == 0:
          print ('Now training epoch %s'%epoch)
      model.train(documents)
      #model.alpha -= 0.002  # decrease the learning rate
      #model.min_alpha = model.alpha  # fix the learning rate, no decay
  
  fname='../atrocitiesdata/d2vmodel.model'
  model.save(fname)
  
  fname='../atrocitiesdata/d2vmodel.model'
  model = gensim.models.Doc2Vec.load(fname)
  
  #print model.vocab
  # shows the similar words
  #print (model.most_similar('BBC'))
  
  # shows the learnt embedding
  #print (model['BBC'])
  
  # shows the similar docs with id = 2
  print (model.docvecs.most_similar(str(2)))
  
  
  '''
  sentences='../atrocitiesdata/ClusterLabel3.txt'
  ClusterLabel=[]
  f = open(sentences, 'r')
  for d in f:
    ClusterLabel.append(int(d.strip()))
  
  sentences='../atrocitiesdata/KeysList4.txt'
  KeysList=[]
  f = open(sentences, 'r')
  for d in f:
    KeysList.append(d.decode('utf-8').strip())
  
  
  print len(ClusterLabel), len(KeysList)
  
  rootdir='../atrocitiesdata/OrderedDocuments/'
  dirsList=next(os.walk(rootdir))[1]
  for d in range(len(dirsList)):
    dirsList[d]= datetime.strptime(dirsList[d], '%d_%b_%Y')
  
  dirsList.sort()
  dirsList=[os.path.join(rootdir, date.strftime(s,'%d_%b_%Y')) for s in dirsList]
  
  nTrainingdays=5
  n=len(dataText)
  #windowsize=3
  #instanceslist=[numpy.zeros((1,5))]*windowsize
  #Labelslist=[[1]]*windowsize
  #windowinstances=[[0]]*windowsize
  AllF=[]
  AllPrecision=[]
  AllRecall=[]
  AllavgF=[]
  AllPurities=[]
  AllAccuracy=[]
  
  Trainingindexes=[]
  chunk=0
  for day in range(len(dirsList)):
    fs=next(os.walk(dirsList[day]))[2]
    for i, j in enumerate(fs):
      key=j.split('.')[0]
      if key in KeysList:
        instanceindex=KeysList.index(key)
        Trainingindexes.append(instanceindex)
    
    if day>5 and day % 5 == 0:
      
      documents =sents_to_taggeddoc([dataText[i] for i in Trainingindexes])
      #data=[]
      #for i in range(len(documents)):
      #  data.append(model.infer_vector(documents[i][0]))
      #
      #datax=numpy.asarray(data)
      '''
      Trmask=numpy.zeros(n,dtype='bool')
      Trmask[Trainingindexes]=True
      datax=data[Trmask,:]
      '''
      clusterslabels=[ClusterLabel[i] for i in Trainingindexes]
      GTclusterslabels=numpy.unique(clusterslabels,return_inverse=True,return_counts=True)
      '''
      data=numpy.hstack((datax,(GTclusterslabels[1]).reshape(len(Trainingindexes),1)))
      data=data.astype(numpy.float)
      header1=["@relation relname"]
      header1.extend(["@attribute a"+str(att) +" numeric" for att in range(data.shape[1]-1)])
      header1.extend(["@attribute class {"+str(map(int,list(numpy.unique(GTclusterslabels[1]))))[1:-1]+"}"])
      header1.extend(["@data\n"])
      numpy.savetxt('../atrocitiesdata/d2vchunks/ch_'+str(chunk)+'.arff', data, fmt='%f,'*(data.shape[1]-1)+'%i', delimiter=',',header="\n".join(header1), comments='')
      chunk+=1
      Trainingindexes=[]
      continue
      #GTclusterslabels[0],GTclusterslabels[1],GTclusterslabels[2]
      kmeans = KMeans(n_clusters=len(GTclusterslabels[0]), random_state=0).fit(datax)
      '''
      fname='../atrocitiesdata/d2vchunks/ch_'+str(chunk)+"_lbl.csv"
      with open(fname,'r') as myfile:
          content=myfile.readlines()
      lbl=map(int,content)
      GTclusterslabelsOrderedIndex=list(numpy.argsort(GTclusterslabels[2]))
      #Predclusterslabels=numpy.unique(kmeans.labels_,return_inverse=True,return_counts=True)
      Predclusterslabels=numpy.unique(lbl,return_inverse=True,return_counts=True)
      PredclusterslabelsOrderedIndex=list(numpy.argsort(Predclusterslabels[2]))
      ncorrect=0
      ConfusionMatrix=numpy.zeros((len(GTclusterslabelsOrderedIndex),len(PredclusterslabelsOrderedIndex)))
      for Predidx,Pred in enumerate(Predclusterslabels[1]):
        if PredclusterslabelsOrderedIndex.index(Pred) ==  GTclusterslabelsOrderedIndex.index(GTclusterslabels[1][Predidx]):
          ncorrect +=1
        ConfusionMatrix[PredclusterslabelsOrderedIndex.index(Pred),GTclusterslabelsOrderedIndex.index(GTclusterslabels[1][Predidx])] += 1
      
      accuracy = 1.0*ncorrect/len(GTclusterslabelsOrderedIndex)
      purity = numpy.sum(numpy.max(ConfusionMatrix,1))/numpy.sum(ConfusionMatrix)
      precisionList = numpy.max(ConfusionMatrix, 1) / numpy.sum(ConfusionMatrix, 1)
      maxidx = numpy.argmax(ConfusionMatrix, 1)
      recallList = ConfusionMatrix[range(ConfusionMatrix.shape[0]),maxidx] / numpy.sum(ConfusionMatrix[:,maxidx],0)
      F1List = 2*precisionList*recallList/(precisionList+recallList)
      
      AllAccuracy.append(accuracy)
      avgF = numpy.average(F1List)
      AllF.append(F1List)
      AllPrecision.append(precisionList)
      AllRecall.append(recallList)
      AllavgF.append(avgF)
      AllPurities.append(purity)
      Trainingindexes=[]
      chunk+=1
  
  print("Done...")
  avgAllacc = numpy.average(AllAccuracy)
  avgAllPurities = numpy.average(AllPurities)
  avgAllF = numpy.average(AllavgF)
  Precision=numpy.average([map(lambda x: numpy.average(x), AllPrecision)])
  Recall=numpy.average([map(lambda x: numpy.average(x), AllRecall)])
  
  
  log = [str(Precision),str(Recall),str(avgAllPurities),str(avgAllF),str(avgAllacc), "\n"]
  print "\t".join(["Precision","Recall","Purity","F1","Accuracy"])
  print "\t".join(log)



