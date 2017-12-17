### import all needed packages here
import bs4
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup

### formal part

## get link and get prepared
my_url='insert target link here'
#opening up connection, grabbing the page
uClient = uReq(my_url)
page_html = uClient.read()
uClient.close()

## html parsing and let computer know the code
page_soup = soup(page_html,"html.parser")

## locate the position for materials that we want
containers = page_soup.findAll("div",{"class":"class name"})

# check the length of containers
len(containers)

## create a file
filename = "inputname.csv"

# w for write text
f = open(filename, "W")

# define column name
headers = "col_name1, col_name2, col_name3\n"

## start writing
f.write(headers)
for container in containers:
    col_name1 = container.div.div.a.img["title"]
    title_container = container.findAll("a", {"class":"item-title"})
    col_name2 = title_container[0].text
    shipping_container = container.findAll("li", {"class":"price-ship"})
    col_name3 = shipping_container[0].text.strip()

    print("col_name1:" + col_name1)
    print("col_name2:" + col_name2)
    print("col_name3:" + col_name3)
    
    f.write(col_name1 + "," + col_name2.replace(",", "|") + "," + col_name3 "\n")
f.close()
