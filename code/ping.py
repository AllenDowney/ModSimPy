
import urllib.request
import urllib.error
from bs4 import BeautifulSoup
import requests



sound_file = './sound/beep.wav'

headers = {'User_Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'}
url ='http://python.ort'

try:
    # headers = {'User_Agent': 'Mozilla/5.0 (X11; Ubuntu; 
    #             Linux x86_64; rv:57.0) Gecko/20100101 
    #             Firefox/57.0'}
    response = urllib.request.Request('http://python.org/', 
                                       headers=headers)
    html = urllib.request.urlopen(response)

    result = html.read().decode('utf-8')

except urllib.error.URLError as e:
    if hasattr(e, 'reason'):
        print('错误原因是' + str(e.reason))
except urllib.error.HTTPError as e:
    if hasattr(e, 'code'):
        print('错误状态码是' + str(e.code))
else:
    print('请求成功通过。')
    # print(result.text)


    # soup = BeautifulSoup(result, 'html.parser')
    # print(soup.prettify())
    # print(len(list(soup.children)))
    # print(len(list(soup.descendants)))


#     print(soup.p)
#     print(soup.a)
#     print(soup.li)
#     print(soup.find_all('a'))
#     print(soup.head)
#     print(result)
# get value href =""

# One common task is extracting all the URLs found within a page’s <a> tags:
# for link in soup.find_all('a'):
#     print(link.get('id'))
#     print(link.get('href'))

# Another common task is extracting all the text from a page:
# print(soup.get_text())





