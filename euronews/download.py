from newspaper import Article,build
import json
from pathlib import Path
import time

arlinks=[]
eslinks=[]
ptlinks=[]
enlinks=[]

falinks=[]
grlinks=[]
hulinks=[]
rulinks=[]
trlinks=[]
itlinks=[]
delinks=[]
frlinks=[]

dates=['08_01/','08_02/','08_03/','08_04/','08_05/','08_06/','07_01/','07_02/','07_03/','07_04/','07_05/','07_06/','07_07/','07_08/','07_09/','07_10/','07_11/','07_12/','07_13/','07_14/','07_15/','07_16/','07_17/','07_18/','07_19/','07_20/','07_21/','07_22/','07_23/','07_24/','07_25/','07_26/','07_27/','07_28/','07_29/','07_30/','07_31/','06_01','06_02','06_03','06_04','06_05','06_06','06_07','06_08','06_09','06_10','06_11','06_12','06_13','06_14','06_15','06_16','06_17','06_18','06_19','06_20','06_21','06_22','06_23','06_24','06_25','06_26','06_27','06_28','06_29','06_30']

for d in dates:
  pathfilename='/home/ahmad/duplicate-detection/euronews/data/2017_'+d
  
  my_file = Path(pathfilename+'/arlink.txt')
  if not my_file.is_file():
    continue
  
  try:
    with open(pathfilename+'/arlink.txt','r') as myfile:
      arlink=myfile.readlines()
    
    with open(pathfilename+'/eslink.txt','r') as myfile:
      eslink=myfile.readlines()
    
    with open(pathfilename+'/ptlink.txt','r') as myfile:
      ptlink=myfile.readlines()
    
    with open(pathfilename+'/enlink.txt','r') as myfile:
      enlink=myfile.readlines()
    
    
    with open(pathfilename+'/falink.txt','r') as myfile:
      falink=myfile.readlines()
    
    with open(pathfilename+'/grlink.txt','r') as myfile:
      grlink=myfile.readlines()
    
    with open(pathfilename+'/hulink.txt','r') as myfile:
      hulink=myfile.readlines()
    
    with open(pathfilename+'/rulink.txt','r') as myfile:
      rulink=myfile.readlines()
    
    with open(pathfilename+'/trlink.txt','r') as myfile:
      trlink=myfile.readlines()
    
    with open(pathfilename+'/itlink.txt','r') as myfile:
      itlink=myfile.readlines()
    
    with open(pathfilename+'/delink.txt','r') as myfile:
      delink=myfile.readlines()
    
    with open(pathfilename+'/frlink.txt','r') as myfile:
      frlink=myfile.readlines()
    
    for lidx,l in enumerate(enlink):
      if any(lidx >= i for i in [len(enlink),len(arlink),len(eslink),len(ptlink),len(falink),len(grlink),len(hulink),len(rulink),len(trlink),len(itlink),len(delink),len(frlink)]):
        break
      
      if l.strip() not in enlinks:
        enlinks.append(enlink[lidx].strip())
        arlinks.append(arlink[lidx].strip())
        eslinks.append(eslink[lidx].strip())
        ptlinks.append(ptlink[lidx].strip())
        falinks.append(falink[lidx].strip())
        grlinks.append(grlink[lidx].strip())
        hulinks.append(hulink[lidx].strip())
        rulinks.append(rulink[lidx].strip())
        trlinks.append(trlink[lidx].strip())
        itlinks.append(itlink[lidx].strip())
        delinks.append(delink[lidx].strip())
        frlinks.append(frlink[lidx].strip())
  
  except:
    pass



print len(enlinks),len(arlinks),len(eslinks),len(ptlinks),len(falinks),len(grlinks),len(hulinks),len(rulinks),len(trlinks),len(itlinks),len(delinks),len(frlinks)


secs=0.5
narticles=0
jsonfiles=[]
for artidx,art in enumerate(enlinks):
  if artidx >= len(arlinks):# or artidx >= len(eslinks) or artidx >=len(ptlinks):
    break
  if enlinks[artidx] == "":
    continue
  if arlinks[artidx] == "":
    continue
  #if eslinks[artidx] == "":
  #  continue
  #if ptlinks[artidx] == "":
  #  continue
  print(artidx,':',art)
  article = Article(art, language='en')
  article.download()
  sec_n=secs
  while article.download_state==0:
    article.download()
    time.sleep(sec_n)
    sec_n += 0.5
  if article.download_state==1:
    continue
  article.parse()
  jsondict={'id':article.publish_date.strftime("%Y%m%d")+'_'+str(narticles).zfill(4), 'URL':{'en':art},'date':{'en':article.publish_date.strftime("%x %X")}, 'title':{'en':article.title}, 'text':{'en':article.text}, 'keywords':{'en':article.keywords}, 'top_image':{'en':article.top_image}}
  
  article = Article(arlinks[artidx], language='ar') # arabic
  article.download()
  sec_n=secs
  while article.download_state==0:
    article.download()
    time.sleep(sec_n)
    sec_n += 0.5
  if article.download_state==1:
    continue
  article.parse()
  jsondict['URL']['ar']=arlinks[artidx]
  jsondict['date']['ar']=article.publish_date.strftime("%x %X")
  jsondict['title']['ar']=article.title
  jsondict['text']['ar']=article.text
  jsondict['keywords']['ar']=article.keywords 
  jsondict['top_image']['ar']=article.top_image
  '''
  article = Article(eslinks[artidx], language='es') # spanish
  article.download()
  sec_n=secs
  while article.download_state==0:
    article.download()
    time.sleep(sec_n)
    sec_n += 0.5
  if article.download_state==1:
    continue
  article.parse()
  jsondict['URL']['es']=eslinks[artidx]
  jsondict['date']['es']=article.publish_date.strftime("%x %X")
  jsondict['title']['es']=article.title
  jsondict['text']['es']=article.text
  jsondict['keywords']['es']=article.keywords 
  jsondict['top_image']['es']=article.top_image
  
  article = Article(ptlinks[artidx], language='pt') # Portuguese
  article.download()
  sec_n=secs
  while article.download_state==0:
    article.download()
    time.sleep(sec_n)
    sec_n += 0.5
  if article.download_state==1:
    continue
  article.parse()
  jsondict['URL']['pt']=ptlinks[artidx]
  jsondict['date']['pt']=article.publish_date.strftime("%x %X")
  jsondict['title']['pt']=article.title
  jsondict['text']['pt']=article.text
  jsondict['keywords']['pt']=article.keywords 
  jsondict['top_image']['pt']=article.top_image
  '''
  #de#fr#it#hu#tr#el#ru
  #with open('data/jsonfiles/'+article.publish_date.strftime("%Y%m%d")+'_'+str(narticles).zfill(4)+'.json', 'w') as myfile:
  #    json.dump(jsondict, myfile)
  jsonfiles.append(jsondict)
  narticles += 1


with open('data/jsonfilecombined/en_ar_20170601_'+str(narticles)+'.json', 'w') as myfile:
  json.dump(jsonfiles, myfile)

print(str(narticles) + " articles")  
'''
url='http://www.euronews.com/2017/08/03'
cnn_paper = build(url)
cnn_paper = build(url)
for article in cnn_paper.articles:
    print(article.url)
    break
print(cnn_paper.size())    
for category in cnn_paper.category_urls():
    print(category)
for feed_url in cnn_paper.feed_urls():
    print(feed_url)    

print(cnn_paper.articles[0].url)
print(cnn_paper.articles[1].url)
print(cnn_paper.articles[1].url)
dir(cnn_paper.articles[1])
parse(cnn_paper.articles[1])
cnn_paper.articles[1].source_url
cnn_paper.size()

a = Article(url, language='en')
parse(a)

url = 'http://www.euronews.com/2016/11/01/trade-on-agenda-for-first-uk-visit-by-colombian-president'
a = Article(url, language='en')
parse(a)
print("ar")
url='http://arabic.euronews.com/2016/11/01/trade-on-agenda-for-first-uk-visit-by-colombian-president'
a = Article(url, language='ar') # arabic
parse(a)
print('es')
url='http://es.euronews.com/2016/11/01/juan-manuel-santos-recibido-con-honores-en-londres'
a = Article(url, language='es') # spanish
parse(a)
url='http://pt.euronews.com/2016/11/01/colombia-quer-reforcar-lacos-comerciais-com-o-reino-unido-apos-brexit'
a = Article(url, language='pt') # Portuguese
parse(a)
url='http://tr.euronews.com/2016/11/01/kolombiya-brexit-sonrasi-icin-ingiltere-ile-isbirligi-ariyor'
'''

print len(enlinks),len(arlinks),len(eslinks),len(ptlinks),len(falinks),len(grlinks),len(hulinks),len(rulinks),len(trlinks),len(itlinks),len(delinks),len(frlinks)
secs=0.5
narticles=0
jsonfiles=[]
for artidx,art in enumerate(enlinks):
  if artidx >= len(arlinks):# or artidx >= len(eslinks) or artidx >=len(ptlinks):
    break
  if enlinks[artidx] == "":
    continue
  if arlinks[artidx] == "":
    continue
  #if eslinks[artidx] == "":
  #  continue
  #if ptlinks[artidx] == "":
  #  continue
  print(artidx,':',art)
  article = Article(art, language='en')
  article.download()
  sec_n=secs
  while article.download_state==0:
    article.download()
    time.sleep(sec_n)
    sec_n += 0.5
  if article.download_state==1:
    continue
  article.parse()
  jsondict={'id':article.publish_date.strftime("%Y%m%d")+'_'+str(narticles).zfill(4), 'URL':{'en':art},'date':{'en':article.publish_date.strftime("%x %X")}, 'title':{'en':article.title}, 'text':{'en':article.text}, 'keywords':{'en':article.keywords}, 'top_image':{'en':article.top_image}}
  
  article = Article(arlinks[artidx], language='ar') # arabic
  article.download()
  sec_n=secs
  while article.download_state==0:
    article.download()
    time.sleep(sec_n)
    sec_n += 0.5
  if article.download_state==1:
    continue
  article.parse()
  jsondict['URL']['ar']=arlinks[artidx]
  jsondict['date']['ar']=article.publish_date.strftime("%x %X")
  jsondict['title']['ar']=article.title
  jsondict['text']['ar']=article.text
  jsondict['keywords']['ar']=article.keywords 
  jsondict['top_image']['ar']=article.top_image
  '''
  article = Article(eslinks[artidx], language='es') # spanish
  article.download()
  sec_n=secs
  while article.download_state==0:
    article.download()
    time.sleep(sec_n)
    sec_n += 0.5
  if article.download_state==1:
    continue
  article.parse()
  jsondict['URL']['es']=eslinks[artidx]
  jsondict['date']['es']=article.publish_date.strftime("%x %X")
  jsondict['title']['es']=article.title
  jsondict['text']['es']=article.text
  jsondict['keywords']['es']=article.keywords 
  jsondict['top_image']['es']=article.top_image
  
  article = Article(ptlinks[artidx], language='pt') # Portuguese
  article.download()
  sec_n=secs
  while article.download_state==0:
    article.download()
    time.sleep(sec_n)
    sec_n += 0.5
  if article.download_state==1:
    continue
  article.parse()
  jsondict['URL']['pt']=ptlinks[artidx]
  jsondict['date']['pt']=article.publish_date.strftime("%x %X")
  jsondict['title']['pt']=article.title
  jsondict['text']['pt']=article.text
  jsondict['keywords']['pt']=article.keywords 
  jsondict['top_image']['pt']=article.top_image
  '''
  #de#fr#it#hu#tr#el#ru
  #with open('data/jsonfiles/'+article.publish_date.strftime("%Y%m%d")+'_'+str(narticles).zfill(4)+'.json', 'w') as myfile:
  #    json.dump(jsondict, myfile)
  jsonfiles.append(jsondict)
  narticles += 1


with open('data/jsonfilecombined/en_ar_20170601_'+str(narticles)+'.json', 'w') as myfile:
  json.dump(jsonfiles, myfile)
