
def create_w2v_pairs(w2vmodel,allposfiles,allnegfiles):
  langcode={"eng":"en","spa":"es","deu":"de","zho":"zh","ita":"it","fra":"fr","rus":"ru","swe":"sv","nld":"nl","tur":"tr","jpn":"ja","por":"pt","ara":"ar","fin":"fi","ron":"ro","kor":"ko","hrv":"hr","tam":"","hun":"hu","slv":"sl","pol":"pl","srp":"sr","cat":"ca","ukr":"uk"}
  #w2vmodel= loadmodel('/home/ahmad/duplicate-detection/multilingual-embedding/fifty_nine.table5.multiCCA.size_40.normalized')
  pairs=[]
  wpairs=[]
  langpairs=[]
  EntJaccardSim=[]
  sentpairs=[]
  labels = []
  pospairs=[]
  poswordpairs=[]
  possentpairs=[]
  poslangpairs=[]
  posEntJaccardSim=[]
  #print("creating positive pairs:")
  for idx,Pfilenm in enumerate(allposfiles):
    try:
      with open(Pfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      lgroup1=[]
      group1=[]
      wgroup1=[]
      sentgroup1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        sentgroup1.append(artcle['title'])
        lgroup1.append(langcode[artcle['lang']])
        continue
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        w2vmatrix1=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower() and word.strip().lower() not in stpwords:
            try:
              if type(word)==type(''):
                word=word.strip().lower().decode('utf-8')
              else:
                word=word.strip().lower()
              
              #w2vmatrix1.append(w2vmodel[word])
              w2vmatrix1.append(w2vmodel[langcode[artcle['lang']]][word])
              wlist.append(word.strip().lower())
            except:
              pass
        
        sentgroup1.append(artcle['title'])
        lgroup1.append(langcode[artcle['lang']])
        group1.append(np.array(w2vmatrix1))
        wgroup1.append(wlist)
      
      lgroup2=[]
      group2=[]
      wgroup2=[]
      sentgroup2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        sentgroup2.append(artcle['title'])
        lgroup2.append(langcode[artcle['lang']])
        continue
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        #wordslist=set(wordslist)
        w2vmatrix2=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower() and word.strip().lower() not in stpwords: #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              if type(word)==type(''):
                word=word.strip().lower().decode('utf-8')
              else:
                word=word.strip().lower()
              
              #w2vmatrix2.append(w2vmodel[word])
              w2vmatrix2.append(w2vmodel[langcode[artcle['lang']]][word])
              wlist.append(word.strip().lower())
            except:
              pass
        
        sentgroup2.append(artcle['title'])
        lgroup2.append(langcode[artcle['lang']])
        group2.append(np.array(w2vmatrix2))
        wgroup2.append(wlist)
      
      
      for x1 in range(len(lgroup1)):
        for x2 in range(len(lgroup2)):
          possentpairs.append([sentgroup1[x1],sentgroup2[x2]])
          poslangpairs.append([lgroup1[x1],lgroup2[x2]])
          continue
          if [lgroup1[x1],lgroup2[x2]] not in [['de','en'],['es','en'],['de','es']]:
            #if [lgroup1[x1],lgroup2[x2]] not in [['en','de'],['de','en'],['es','en'],['en','es']]:
            #if lgroup1[x1] not in ['en','es','de'] or lgroup2[x2] not in ['de','es','en']:
            #if lgroup1[x1] not in ['en','es'] or lgroup2[x2] not in ['es','en']:
            continue
          
          
          pospairs.append([np.array(group1[x1]),np.array(group2[x2])])
          poswordpairs.append([wgroup1[x1],wgroup2[x2]])
          poslangpairs.append([lgroup1[x1],lgroup2[x2]])
          posEntJaccardSim.append(jsonfile['meta']['entityJaccardSim'])
          possentpairs.append([sentgroup1[x1],sentgroup2[x2]])
      
      
      
      #sys.stdout.write("\r")
      #sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allposfiles)), 100*idx/len(allposfiles)))
      #sys.stdout.flush()
      
    except:
      pass
  
  
  #print('\ncreating negative pairs...')
  
  neglangpairs=[]
  negpairs=[]
  negwordpairs=[]
  negEntJaccardSim=[]
  negsentpairs=[]
  for idx,Nfilenm in enumerate(allnegfiles):
    try:
      with open(Nfilenm,"r") as myfile:
        jsonfile=json.load(myfile)
      
      keys = [x for x in list(jsonfile.keys()) if "-" in x]
      
      lgroup1=[]
      sentgroup1=[]
      group1=[]
      wgroup1=[]
      for artcle in jsonfile[keys[0]]['articles']['results']:
        sentgroup1.append(artcle['title'])
        lgroup1.append(langcode[artcle['lang']])
        continue
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        #wordslist=set(wordslist)
        w2vmatrix1=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower() and word.strip().lower() not in stpwords: #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and 
            try:
              if type(word)==type(''):
                word=word.strip().lower().decode('utf-8')
              else:
                word=word.strip().lower()
              
              #w2vmatrix1.append(w2vmodel[word])
              w2vmatrix1.append(w2vmodel[langcode[artcle['lang']]][word])
              wlist.append(word)
            except:
              pass
        
        sentgroup1.append(artcle['title'])
        lgroup1.append(langcode[artcle['lang']])
        group1.append(np.array(w2vmatrix1))
        wgroup1.append(wlist)
      
      
      
      lgroup2=[]
      group2=[]
      wgroup2=[]
      sentgroup2=[]
      for artcle in jsonfile[keys[1]]['articles']['results']:
        lgroup2.append(langcode[artcle['lang']])
        sentgroup2.append(artcle['title'])
        continue
        wordslist=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>? ]', artcle['body'])
        #wordslist=set(wordslist)
        w2vmatrix2=[]
        wlist=[]
        for word in wordslist:
          if '' !=word.strip().lower() and word.strip().lower() not in stpwords: #word.strip().lower() in w2vmodel[langcode[artcle['lang']]].keys() and
            try:
              if type(word)==type(''):
                word=word.strip().lower().decode('utf-8')
              else:
                word=word.strip().lower()
              
              #w2vmatrix2.append(w2vmodel[word])
              w2vmatrix2.append(w2vmodel[langcode[artcle['lang']]][word])
              wlist.append(word.strip().lower())
            except:
              pass
        
        sentgroup2.append(artcle['body'])
        lgroup2.append(langcode[artcle['lang']])
        group2.append(np.array(w2vmatrix2))
        wgroup2.append(wlist)
      
      for x1 in range(len(lgroup1)):
        for x2 in range(len(lgroup2)):
          neglangpairs.append([lgroup1[x1],lgroup2[x2]])
          negsentpairs.append([sentgroup1[x1],sentgroup2[x2]])
          continue
          if [lgroup1[x1],lgroup2[x2]] not in [['de','en'],['es','en'],['de','es']]:
            #if [lgroup1[x1],lgroup2[x2]] in [['en','de'],['de','en'],['es','en'],['en','es']]:
            #if lgroup1[x1] not in ['en','es','de'] or lgroup2[x2] not in ['de','es','en']:
            #if lgroup1[x1] not in ['en','es'] or lgroup2[x2] not in ['es','en']:
            continue
          
          
          negpairs.append([np.array(group1[x1]),np.array(group2[x2])])
          negwordpairs.append([wgroup1[x1],wgroup2[x2]])
          neglangpairs.append([lgroup1[x1],lgroup2[x2]])
          negEntJaccardSim.append(jsonfile['meta']['entityJaccardSim'])
          negsentpairs.append([sentgroup1[x1],sentgroup2[x2]])
      
      
      #sys.stdout.write("\r")
      #sys.stdout.write("[%-100s] %d%%" % ('='*(100*idx/len(allnegfiles)), 100*idx/len(allnegfiles)))
      #sys.stdout.flush()
    except:
      pass
  
  
  #print("\nShuffling...")
  #print len(pospairs),len(negpairs),len(poswordpairs),len(negwordpairs),
  #print len(poslangpairs),len(neglangpairs),len(possentpairs),len(negsentpairs)
  for posspair,poslpair in zip(possentpairs,poslangpairs):
    labels.append(1)
    langpairs.append(poslpair)
    sentpairs.append(posspair)
  
  for negspair,neglpair in zip(negsentpairs,neglangpairs):
    labels.append(0)
    langpairs.append(neglpair)
    sentpairs.append(negspair)
  
  return labels, langpairs,sentpairs


stpwords=set(stopwords.words("spanish")+stopwords.words("english")+stopwords.words("german")+stopwords.words("french"))

unifiedw2vmodel=dict()
Allsentpairs=[]
Alllangpairs=[]
Allisdup_labels=[]
posfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/positive/'
negfolderpath='/home/ahmad/duplicate-detection/eventregistrydata/negative/'
posfilenames = [join(posfolderpath, f) for f in listdir(posfolderpath) if isfile(join(posfolderpath, f))]
negfilenames = [join(negfolderpath, f) for f in listdir(negfolderpath) if isfile(join(negfolderpath, f))]
to=min(len(posfilenames),len(negfilenames))
print to
frm=0
#cnt=0
for frm in range(to):
  labels,langpairs,sentpairs=create_w2v_pairs(unifiedw2vmodel,[posfilenames[frm]],[negfilenames[frm]])
  if len(labels)==0:
    continue
  
  #print "processing ",frm,len(w2vpairs),len(w2vpairs[0]), " pairs"
  if frm%50 == 0 and frm>0:
    print frm
    
  Allisdup_labels.extend(labels)
  Alllangpairs.extend(langpairs)
  Allsentpairs.extend(sentpairs)

len(Alllangpairs),len(Allsentpairs)

transallleft=[]
transallright=[]
label=[]
lng=[]

for _i in range(27296,len(Allsentpairs)):
  if len(transallleft) > 15000: #out of 124,948
    print "NEXT", _i
    break
  
  if _i % 1000==0:
    print _i, len(transallleft)
  
  _sent=Allsentpairs[_i]
  try:
    #if ((Alllangpairs[_i][0]=='es' or Alllangpairs[_i][0]=='de') and Alllangpairs[_i][1]=='en') or (Alllangpairs[_i][0]=='de' and Alllangpairs[_i][1]=='es'):
    if (Alllangpairs[_i][0]=='de' and Alllangpairs[_i][0]=='es'):
      translation=service.translations().list(source=Alllangpairs[_i][0],target='en',q=[_sent[0]],format='text').execute()
      #transwords=[transw['translatedText'].encode('utf-8') for transw in translation['translations']]
      transwordsl=translation['translations'][0]['translatedText'].encode('utf-8')
      
      
      if Alllangpairs[_i][1]!='en':
        translation=service.translations().list(source=Alllangpairs[_i][1],target='en',q=_sent[1],format='text').execute()
        #transwords=[transw['translatedText'].encode('utf-8') for transw in translation['translations']]
        transwordsr=translation['translations'][0]['translatedText'].encode('utf-8')
      else:
        transwordsr=_sent[1]
      
      transallleft.append(transwordsl)
      transallright.append(transwordsr)
      
      label.append(Allisdup_labels[_i])
      lng.append(Alllangpairs[_i])
  except:
    print sys.exc_info()[0]
    pass


with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/leftSentgoogleTransenesde.txt', 'wb') as myfile:
      myfile.write("\n".join(transallleft))

for idx in range(len(transallright)):
  transallright[idx]=transallright[idx].encode('utf-8')

with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/rightSentgoogleTransenesde.txt', 'wb') as myfile:
      myfile.write("\n".join(transallright))

with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/lblgoogleTransenesde.txt', 'wb') as myfile:
      myfile.write("\n".join(map(str,label)))

lng0=[_lng[0] for _lng in lng] 
with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/leftlnggoogleTransenesde.txt', 'wb') as myfile:
      myfile.write("\n".join(lng0))

lng1=[_lng[1] for _lng in lng]
with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/rightlnggoogleTransenesde.txt', 'wb') as myfile:
      myfile.write("\n".join(lng1))

import nltk
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item).lower())
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    #stems = stem_tokens(tokens, stemmer)
    return tokens



dataText=transallleft+transallright
dataText_vectorizer = TfidfVectorizer(analyzer = "word",tokenizer = tokenize, preprocessor = None, max_features = 1000) #,ngram_range=(1, 1)
dataText_features = dataText_vectorizer.fit_transform(dataText)
dataText_features = dataText_features.toarray()
n=len(dataText)
leftdataText_features=dataText_features[:n/2,]
rightdataText_features=dataText_features[n/2:,]
cosdistance=[]
for _idx in range(leftdataText_features.shape[0]):
  cosdistance.append(spatial.distance.cosine(leftdataText_features[_idx,],rightdataText_features[_idx,]))

with open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/cosgoogleTransenesde.txt', 'wb') as myfile:
      myfile.write("\n".join(cosdistance))

_lng0='de'
_lng1='es'
countpos=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==1 and _lng[0]==_lng0 and _lng[1]==_lng1])
countneg=sum([True for _lbl,_lng in zip(Allisdup_labels,Alllangpairs) if _lbl==0 and _lng[0]==_lng0 and _lng[1]==_lng1])


h=0.9
TP=sum([True for pp,_lbl,_lng in zip(cosdistance,label,lng) if pp<=h and pp>=0 and _lbl==1 and _lng[0]== _lng0 and _lng[1]== _lng1])
FP=sum([True for pp,_lbl,_lng in zip(cosdistance,label,lng) if pp<=h and pp>=0 and _lbl==0 and _lng[0]==_lng0 and _lng[1]==_lng1])
TN=sum([True for pp,_lbl,_lng in zip(cosdistance,label,lng) if pp>h and pp>=0 and _lbl==0 and _lng[0]==_lng0 and _lng[1]==_lng1])
FN=sum([True for pp,_lbl,_lng in zip(cosdistance,label,lng) if pp>h and pp>=0 and _lbl==1 and _lng[0]==_lng0 and _lng[1]==_lng1])


d=countpos-(TP+FN)
FN+=d
d=countneg-(TN+FP)
FP+=d


h=0.878
TP=sum([True for pp,_lbl,_lng in zip(cosdistance,label,lng) if pp<=h and pp>=0 and _lbl==1 and _lng[0]== _lng0 and _lng[1]== _lng1])
FP=sum([True for pp,_lbl,_lng in zip(cosdistance,label,lng) if pp<=h and pp>=0 and _lbl==0 and _lng[0]==_lng0 and _lng[1]==_lng1])
TN=sum([True for pp,_lbl,_lng in zip(cosdistance,label,lng) if pp>h and pp>=0 and _lbl==0 and _lng[0]==_lng0 and _lng[1]==_lng1])
FN=sum([True for pp,_lbl,_lng in zip(cosdistance,label,lng) if pp>h and pp>=0 and _lbl==1 and _lng[0]==_lng0 and _lng[1]==_lng1])


poserror=sum([True for pp,_lbl in zip(Allpureclustersratio,labels) if pp<0 and _lbl==1])
negerror=sum([True for pp,_lbl in zip(Allpureclustersratio,labels) if pp<0 and _lbl==0])
Precision=100.0*TP/(TP+FP+0.000001)
Recall=100.0*TP/(TP+FN+0.000001)
F1=100.0*(2.0*TP)/((2.0*TP+1.0*FN+FP)+0.000001)
F2=100.0*(5.0*TP)/((5.0*TP+4.0*FN+FP)+0.000001)
print h,'de','es',TP,TN,FP,FN,str(100.0*(TP+TN)/(TP+TN+FP+FN+0.0001)) + "," + str(F1)+", "+ str(F2)+", "+ str(Precision)+", "+ str(Recall)+ str(", ")+ str(100.0*TP/(TP+FN+0.0001))+ str(", ")+ str(100.0*TN/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FP/(TN+FP+0.0001))+ str(", ")+ str( 100.0*FN/(TP+FN+0.0001))+", "+str((TP+FN))+", "+str((TN+FP))+", "+str((1.0*TP+FN)/(TN+FP+TP+FN+0.0001)),h,poserror,negerror

print Precision,Recall
print 100.0*FP/(FP+TN),100.0*FN/(FN+TP)

TP+TN+FP+FN

len(Alllangpairs),len(cosdistance),len(Allsentpairs),len(Allisdup_labels)

