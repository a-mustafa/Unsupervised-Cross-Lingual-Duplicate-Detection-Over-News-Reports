
import urllib, json, urllib2
from contextlib import closing


def CallWikifier(text, lang="en", threshold=0.8):
    data = urllib.urlencode({"text": text,"lang": lang,"userKey": "qotdbzmynehdoatxunbyxylhagxarn","pageRankSqThreshold": "%g" % threshold,"applyPageRankSqThreshold": "true","nTopDfValuesToIgnore": "200","wikiDataClasses": "true","wikiDataClassIds": "false","support": "true","ranges": "false","includeCosines": "false","maxMentionEntropy": "3"})
    url = "http://www.wikifier.org/annotate-article"
    # Call the Wikifier and read the response.
    data = data.encode('utf8')
    req = urllib2.Request(url, data)
    with closing(urllib2.urlopen(req)) as response:
        response = response.read()
        response = json.loads(response.decode("utf8"))
    
    return [annotation["title"] for annotation in response["annotations"]]
    #print([annotation["title"] for annotation in response["annotations"]])
    # Output the annotations.
    #for annotation in response["annotations"]:
    #    print("%s (%s)" % (annotation["title"], annotation["url"]))

#len(Alllangpairs)= 591420
#CallWikifier(Allsentpairs[i-1][1].encode('utf-8'), lang=Alllangpairs[i-1][1])

n=len(rightNE) #8862
#leftNE=[]
#rightNE=[]
for i,artcl,lbl,lng in zip(range(len(Allsentpairs)),Allsentpairs,Allisdup_labels,Alllangpairs):
  if i < n:
    continue
  
  try:
    NElist=CallWikifier(artcl[0].encode('utf-8'), lang=lng[0])
  except Exception as e:
    print str(e)
    NElist=[]
  
  if i<len(leftNE):
    leftNE[i]=list(set(NElist))
  else:
    leftNE.append(list(set(NElist)))
  
  try:
    NElist=CallWikifier(artcl[1].encode('utf-8'), lang=lng[1])
  except Exception as e:
    print str(e)
    NElist=[]
  
  if i<len(rightNE):
    rightNE[i]=list(set(NElist))
  else:
    rightNE.append(list(set(NElist)))
  
  if i%500==0:
    print i

print len(leftNE), len(rightNE)
CallWikifier("Syria's foreign minister has said Damascus is ready to offer a prisoner exchange with rebels.")

text=Allsentpairs[i][0].encode('utf-8')
lang=Alllangpairs[i][0]
threshold=0.8
data = urllib.urlencode({"text": text,"lang": lang,"userKey": "qotdbzmynehdoatxunbyxylhagxarn","pageRankSqThreshold": "%g" % threshold,"applyPageRankSqThreshold": "true","nTopDfValuesToIgnore": "200","wikiDataClasses": "true","wikiDataClassIds": "false","support": "true","ranges": "false","includeCosines": "false","maxMentionEntropy": "3"})
data = data.encode('utf8')
url = "http://www.wikifier.org/annotate-article"
req = urllib2.Request(url, data)
with closing(urllib2.urlopen(req)) as response:
    response = response.read()
    response = json.loads(response.decode("utf8"))

for ii in range(len(response["annotations"])):
  if 'Sunni' in response["annotations"][ii]['title']:
    print ii
    break

