
import urllib.request
import urllib.error


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


# response = urllib.request.urlopen(url)
# result = response.getcode()
# # print(result)
# # if result ==200:  #url is working
# # 	result =  response.read().decode('utf-8')
# # elif result ==503:
# # 	print('erros')
# # else:
# # 	print(erros)
# # print(result)
# except urllib.error.URLError as e:
#     if hasattr(e, 'reason'):
#         print('错误原因是' + str(e.reason))
# except urllib.error.HTTPError as e:
#     if hasattr(e, 'code'):
#         print('错误状态码是' + str(e.code))
# else:
#     print('请求成功通过。')