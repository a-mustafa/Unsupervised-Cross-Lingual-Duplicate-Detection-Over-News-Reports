import urllib2
#from bs4 import BeautifulSoup
from BeautifulSoup import BeautifulSoup
import os
#urls=['http://www.euronews.com/2017/08/05','http://www.euronews.com/2017/08/06','http://www.euronews.com/2017/08/02',
#urls=['http://www.euronews.com/2017/08/01','http://www.euronews.com/2017/07/31','http://www.euronews.com/2017/07/30','http://www.euronews.com/2017/07/29','http://www.euronews.com/2017/07/28','http://www.euronews.com/2017/07/27','http://www.euronews.com/2017/07/26','http://www.euronews.com/2017/07/25','http://www.euronews.com/2017/07/24','http://www.euronews.com/2017/07/23','http://www.euronews.com/2017/07/22','http://www.euronews.com/2017/07/21','http://www.euronews.com/2017/07/20','http://www.euronews.com/2017/07/19','http://www.euronews.com/2017/07/14','http://www.euronews.com/2017/07/13','http://www.euronews.com/2017/07/12','http://www.euronews.com/2017/07/11','http://www.euronews.com/2017/07/10','http://www.euronews.com/2017/07/09','http://www.euronews.com/2017/07/08','http://www.euronews.com/2017/07/07','http://www.euronews.com/2017/07/06','http://www.euronews.com/2017/07/05','http://www.euronews.com/2017/07/04','http://www.euronews.com/2017/07/03','http://www.euronews.com/2017/07/02','http://www.euronews.com/2017/07/01']

#urls=['http://www.euronews.com/2017/06/01','http://www.euronews.com/2017/06/02','http://www.euronews.com/2017/06/03','http://www.euronews.com/2017/06/04','http://www.euronews.com/2017/06/05','http://www.euronews.com/2017/06/06','http://www.euronews.com/2017/06/07','http://www.euronews.com/2017/06/08','http://www.euronews.com/2017/06/09','http://www.euronews.com/2017/06/10','http://www.euronews.com/2017/06/11','http://www.euronews.com/2017/06/12','http://www.euronews.com/2017/06/13','http://www.euronews.com/2017/06/14','http://www.euronews.com/2017/06/15','http://www.euronews.com/2017/06/16','http://www.euronews.com/2017/06/17','http://www.euronews.com/2017/06/18','http://www.euronews.com/2017/06/19','http://www.euronews.com/2017/06/20','http://www.euronews.com/2017/06/21','http://www.euronews.com/2017/06/22','http://www.euronews.com/2017/06/23','http://www.euronews.com/2017/06/24','http://www.euronews.com/2017/06/25','http://www.euronews.com/2017/06/26','http://www.euronews.com/2017/06/27',
#urls=['http://www.euronews.com/2017/06/28','http://www.euronews.com/2017/06/29','http://www.euronews.com/2017/06/30']
urls=['http://www.euronews.com/2017/08/07','http://www.euronews.com/2017/08/08','http://www.euronews.com/2017/08/09','http://www.euronews.com/2017/08/10','http://www.euronews.com/2017/08/11','http://www.euronews.com/2017/08/12','http://www.euronews.com/2017/08/13','http://www.euronews.com/2017/08/14','http://www.euronews.com/2017/08/15','http://www.euronews.com/2017/08/16','http://www.euronews.com/2017/08/17','http://www.euronews.com/2017/08/18','http://www.euronews.com/2017/08/19','http://www.euronews.com/2017/08/20','http://www.euronews.com/2017/08/21','http://www.euronews.com/2017/08/22','http://www.euronews.com/2017/08/23','http://www.euronews.com/2017/08/24','http://www.euronews.com/2017/08/25','http://www.euronews.com/2017/08/26','http://www.euronews.com/2017/08/27','http://www.euronews.com/2017/08/28','http://www.euronews.com/2017/08/29','http://www.euronews.com/2017/08/30','http://www.euronews.com/2017/08/31','http://www.euronews.com/2017/09/30','http://www.euronews.com/2017/09/29','http://www.euronews.com/2017/09/28','http://www.euronews.com/2017/09/27','http://www.euronews.com/2017/09/26','http://www.euronews.com/2017/09/25','http://www.euronews.com/2017/09/24','http://www.euronews.com/2017/09/23','http://www.euronews.com/2017/09/22','http://www.euronews.com/2017/09/21','http://www.euronews.com/2017/09/20','http://www.euronews.com/2017/09/19','http://www.euronews.com/2017/09/18','http://www.euronews.com/2017/09/17','http://www.euronews.com/2017/09/16','http://www.euronews.com/2017/09/15','http://www.euronews.com/2017/09/14','http://www.euronews.com/2017/09/13','http://www.euronews.com/2017/09/12','http://www.euronews.com/2017/09/11','http://www.euronews.com/2017/09/10','http://www.euronews.com/2017/09/09','http://www.euronews.com/2017/09/08','http://www.euronews.com/2017/09/07','http://www.euronews.com/2017/09/06','http://www.euronews.com/2017/09/05','http://www.euronews.com/2017/09/04','http://www.euronews.com/2017/09/03','http://www.euronews.com/2017/09/02','http://www.euronews.com/2017/09/01','http://www.euronews.com/2017/10/18','http://www.euronews.com/2017/10/17','http://www.euronews.com/2017/10/16','http://www.euronews.com/2017/10/15','http://www.euronews.com/2017/10/14','http://www.euronews.com/2017/10/13','http://www.euronews.com/2017/10/12','http://www.euronews.com/2017/10/11','http://www.euronews.com/2017/10/10','http://www.euronews.com/2017/10/09','http://www.euronews.com/2017/10/08','http://www.euronews.com/2017/10/07','http://www.euronews.com/2017/10/06','http://www.euronews.com/2017/10/05','http://www.euronews.com/2017/10/04','http://www.euronews.com/2017/10/03','http://www.euronews.com/2017/10/02','http://www.euronews.com/2017/10/01']
for url in urls:
  print(url)
  conn = urllib2.urlopen(url)
  html = conn.read()
  soup = BeautifulSoup(html)
  
  listout=[]
  for link in soup.findAll('a', {'class': 'media__body__link'}):
    try:
        listout.append(link['href'])
    except KeyError:
        pass
  
  arlink=[]
  falink=[]
  grlink=[]
  hulink=[]
  eslink=[]
  ptlink=[]
  rulink=[]
  trlink=[]
  itlink=[]
  delink=[]
  frlink=[]
  enlink=[]
  
  for li in listout:
    if 'euronews.com' in li:
      continue
    enurl='http://www.euronews.com'+li
    conn = urllib2.urlopen(enurl)
    html = conn.read()
    soup = BeautifulSoup(html)
    arurl=soup.find('a', {'lang':"hy-AM"},href=True)['href']
    if arurl not in arlink and "-" in arurl: 
      arlink.append(arurl)
    else:
      arlink.append('')
    
    faurl=soup.find('a', {'lang':"fa-IR"},href=True)['href']
    if faurl not in falink and "-" in faurl:
      falink.append(faurl)
    else:
      falink.append('')
    
    grurl=soup.find('a', {'lang':"el-GR"},href=True)['href']
    if grurl not in grlink  and "-" in grurl:
      grlink.append(grurl)
    else:
      grlink.append('')
    
    huurl=soup.find('a', {'lang':"hu-HU"},href=True)['href']
    if huurl not in hulink and "-" in huurl:
      hulink.append(huurl)
    else:
      hulink.append('')
    
    esurl=soup.find('a', {'lang':"es-ES"},href=True)['href']
    if esurl not in eslink and "-" in esurl:
      eslink.append(esurl)
    else:
      eslink.append('')
    
    pturl=soup.find('a', {'lang':"pt-PT"},href=True)['href']
    if pturl not in ptlink and "-" in pturl:
      ptlink.append(pturl)
    else:
      ptlink.append('')
      
    ruurl=soup.find('a', {'lang':"ru-RU"},href=True)['href']
    if ruurl not in rulink and "-" in ruurl:
      rulink.append(ruurl)
    else:
      rulink.append('')
      
    trurl=soup.find('a', {'lang':"tr-TR"},href=True)['href']
    if trurl not in trlink and "-" in trurl:
      trlink.append(trurl)
    else:
      trlink.append('')
    
    iturl=soup.find('a', {'lang':"it-IT"},href=True)['href']
    if iturl not in itlink and "-" in iturl:
      itlink.append(iturl)
    else:
      itlink.append('')
    
    deurl=soup.find('a', {'lang':"de-DEU"},href=True)['href']
    if deurl not in delink and "-" in deurl:
      delink.append(deurl)
    else:
      delink.append('')
    
    frurl = soup.find('a', {'lang':"fr-FR"},href=True)['href']
    if frurl not in frlink and "-" in frurl:
      frlink.append(frurl)
    else:
      frlink.append('')
    
    if enurl not in enlink and "-" in enurl:
      enlink.append(enurl)
    else:
      enlink.append('')
  
  pathfilename='/home/ahmad/duplicate-detection/euronews/data/'+url[url.index('2017'):].replace("/","_")+'/'
  if not os.path.exists(os.path.dirname(pathfilename)):
      os.makedirs(os.path.dirname(pathfilename))
  
  with open(pathfilename+'arlink.txt','w') as myfile:
    myfile.write("\n".join(arlink))
  
  with open(pathfilename+'falink.txt','w') as myfile:
    myfile.write("\n".join(falink))
  
  with open(pathfilename+'grlink.txt','w') as myfile:
    myfile.write("\n".join(grlink))
  
  with open(pathfilename+'hulink.txt','w') as myfile:
    myfile.write("\n".join(hulink))
  
  with open(pathfilename+'eslink.txt','w') as myfile:
    myfile.write("\n".join(eslink))
  
  with open(pathfilename+'ptlink.txt','w') as myfile:
    myfile.write("\n".join(ptlink))
  
  with open(pathfilename+'rulink.txt','w') as myfile:
    myfile.write("\n".join(rulink))
  
  with open(pathfilename+'trlink.txt','w') as myfile:
    myfile.write("\n".join(trlink))
  
  with open(pathfilename+'itlink.txt','w') as myfile:
    myfile.write("\n".join(itlink))
  
  with open(pathfilename+'delink.txt','w') as myfile:
    myfile.write("\n".join(delink))
  
  with open(pathfilename+'frlink.txt','w') as myfile:
    myfile.write("\n".join(frlink))
  
  with open(pathfilename+'enlink.txt','w') as myfile:
    myfile.write("\n".join(enlink))
