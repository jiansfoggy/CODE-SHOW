### import needed packages and get targeted link
import bs4
import random
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
my_url='insert link here'

### formal part
## deal with the link and get prepared, opening up connection, grabbing the page
uClient = uReq(my_url)
page_html = uClient.read()
uClient.close()

## html parsing and get code
page_soup = soup(page_html,"html.parser")

# in some page, they may use other word instead of img to represent pictures
containers = page_soup.findAll("img")
len(containers)

## write download function here
i=1
import urllib.request
def downloadit (pic_link,pic_name):
    # we can name it use random number
    #name=random.randrange(1,1000)
    full_name = str(pic_name) + ".jpg"
    urllib.request.urlretrieve(pic_link,full_name)

## start download here
for pic in containers:
    # get picture link
    temp1=pic.get('src')
    # get picture's name
    temp2=pic.get('alt')
    downloadit (temp1,temp2)

## testing part
    #imagefile = open(nametemp + ".jpeg", 'wb')
    #imagefile.write(uReq(image).read())
    #imagefile.close()
    #print(pic.get('src'))
    #print(pic.get('alt'))
