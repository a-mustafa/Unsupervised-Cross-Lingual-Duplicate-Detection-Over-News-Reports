from newspaper import Article,build
import json
from pathlib import Path
import time
from random import choice
import string
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

directorypath='/home/ahmad/duplicate-detection/euronews/data/'
dates=['08_01/','08_02/','08_03/','08_04/','08_05/','08_06/','07_01/','07_02/','07_03/','07_04/','07_05/','07_06/','07_07/','07_08/','07_09/','07_10/','07_11/','07_12/','07_13/','07_14/','07_15/','07_16/','07_17/','07_18/','07_19/','07_20/','07_21/','07_22/','07_23/','07_24/','07_25/','07_26/','07_27/','07_28/','07_29/','07_30/','07_31/','06_01/','06_02/','06_03/','06_04/','06_05/','06_06/','06_07/','06_08/','06_09/','06_10/','06_11/','06_12/','06_13/','06_14/','06_15/','06_16/','06_17/','06_18/','06_19/','06_20/','06_21/','06_22/','06_23/','06_24/','06_25/','06_26/','06_27/','06_28/','06_29/','06_30/']
dates=dates[37:]
dates=['08_07/','08_08/','08_09/','08_10/','08_11/','08_12/','08_13/','08_14/','08_15/','08_16/','08_17/','08_18/','08_19/','08_20/','08_21/','08_22/','08_23/','08_24/','08_25/','08_26/','08_27/','08_28/','08_29/','08_30/','08_31/','09_30/','09_29/','09_28/','09_27/','09_26/','09_25/','09_24/','09_23/','09_22/','09_21/','09_20/','09_19/','09_18/','09_17/','09_16/','09_15/','09_14/','09_13/','09_12/','09_11/','09_10/','09_09/','09_08/','09_07/','09_06/','09_05/','09_04/','09_03/','09_02/','09_01/','10_18/','10_17/','10_16/','10_15/','10_14/','10_13/','10_12/','10_11/','10_10/','10_09/','10_08/','10_07/','10_06/','10_05/','10_04/','10_03/','10_02/','10_01/']

for d in dates:
  pathfilename=directorypath+'2017_'+d
  
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
    
    print(d)
    jsonfiles=[]
    for lidx,l in enumerate(enlink):
      if all([lidx < i for i in [len(enlink),len(arlink),len(eslink),len(ptlink),len(falink),len(grlink),len(hulink),len(rulink),len(trlink),len(itlink),len(delink),len(frlink)]]):
        if l.strip() not in enlinks:
          links={'en':enlink[lidx].strip(),'ar':arlink[lidx].strip(),'es':eslink[lidx].strip(),'pt':ptlink[lidx].strip(),'fa':falink[lidx].strip(),'gr':grlink[lidx].strip(),'hu':hulink[lidx].strip(),'ru':rulink[lidx].strip(),'tr':trlink[lidx].strip(),'it':itlink[lidx].strip(),'de':delink[lidx].strip(),'fr':frlink[lidx].strip()}
          if sum([len(_links)>0 for _links in links.values()])>=2:
            jsondict=download(links)
            enlinks.append(l.strip())
            jsonfiles.append(jsondict)
    
    
    
    if len(jsonfiles)>0:
      with open(directorypath+'jsonfiles2/'+'2017_'+d[:-1]+'.json', 'w') as myfile:
          json.dump(jsonfiles, myfile)
  
  except:
    pass




def download(links=dict()):
  secs=0.5
  jsondict=dict()
  for lng in links.keys():
    if 'euronews' not in links[lng]:
      continue
    
    if lng=='fa':
      article = Article(links[lng], language='ar')
    elif lng=='gr':
      article = Article(links[lng])
    else:
      article = Article(links[lng], language=lng)
    
    article.download()
    sec_n=secs
    while article.download_state==0:
      article.download()
      time.sleep(sec_n)
      sec_n += 0.5
    if article.download_state==1:
      continue
    article.parse()
    jsondict[lng]={'URL':links[lng],'date':article.publish_date.strftime("%x %X"), 'title':article.title, 'text':article.text, 'keywords':article.keywords, 'top_image':article.top_image}
    
  
  if article is not None:
    jsondict['id']=article.publish_date.strftime("%Y%m%d")+'_'+''.join(choice(string.ascii_uppercase + string.digits) for _ in range(10))
  
  return jsondict



'''
, language='es'
, language='ar'

# import libraries
import urllib2
from bs4 import BeautifulSoup

quote_page = 'http://fa.euronews.com/2017/08/01/what-s-flat-yet-stuffed-touristy-yet-tasty'
quote_page = 'http://fa.euronews.com/2017/10/18/iran-supreme-security-council-head-itw'
page = urllib2.urlopen(quote_page)
soup = BeautifulSoup(page, 'html.parser')
news_box = soup.find('div', attrs={'class': 'c-article-content js-article-content article__content selectionShareable'})
news_box.text
mydivs = soup.find_all("div", { "class" : 'c-article-content js-article-content article__content selectionShareable' })
soup.find_all("div", class_='c-article-content js-article-content article__content selectionShareable')

hit = 
cnt=0
for hit in soup.find_all(name='div'):
    if 'article' in str(hit) and 'content' in str(hit):
      i=str(hit).index('c-article-content')
      print str(hit)[i-50:i+100]
      break
      cnt+=1
for des in hit.descendants:
  des.find_all(name='div')

for child in hit.findChildren():
  if 'article' in str(child) and 'content' in str(child):
    i=str(child).index('c-article-content')
    print str(child)[i-50:i+100]
    break
          i=str(child).index('c-article-content')
          print str(child)[i-50:i+100]
          break
      break




for hit1 in hit.find_all(name="div"):
  if 'c-article-content' in str(hit1):
    break

hit.find_all(name="div", class_='c-article-content js-article-content article__content selectionShareable')



    #de#fr#it#hu#tr#el#ru
    #with open('data/jsonfiles/'+article.publish_date.strftime("%Y%m%d")+'_'+str(narticles).zfill(4)+'.json', 'w') as myfile:
    #    json.dump(jsondict, myfile)
    #jsonfiles.append(jsondict)
    #narticles += 1
  with open('data/jsonfilecombined/en_ar_20170601_'+str(narticles)+'.json', 'w') as myfile:
    json.dump(jsonfiles, myfile)

print(str(narticles) + " articles")
'''