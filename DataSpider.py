#coding:utf-8
import requests
import codecs
import re
import json


class DataSpider:
    def get_data(self, score=0, filename='haha'):
        session = requests.Session()
        session.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)'
                                         'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        with codecs.open(filename, 'w', 'utf-8') as file:
            for page in range(1000):
                try:
                    url = self.get_URL(score, page)
                    data = session.get(url)
                    data = re.sub(r'fetchJSON_comment98vv37157\(', '', data.text)
                    data = data[:-2]
                    data = json.loads(data)
                    for each in data['comments']:
                        file.write(each['content'].strip('\n') + '\n')
                    print(url)
                    print('Finished!')
                except:
                    print('error')
                    pass

    def get_URL(self, score=0, page=0):
        url = 'https://club.jd.com/comment/productPageComments.action?callback=' \
              'fetchJSON_comment98vv37157&productId=4675696&score=' + str(score) + \
              '&sortType=6&page=' + str(page) + '&pageSize=10&isShadowSku=0&fold=1'
        return url
