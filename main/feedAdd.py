# coding=utf8
import json,os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import requests

from bs4 import BeautifulSoup
from app_freshrss.feedParse import get_entry_list

# 读取环境变量 - 数据库
mysql_host = os.getenv('base_mysql_host')
mysql_port = int(os.getenv('base_mysql_port'))
mysql_user = os.getenv('base_mysql_user')
mysql_pwd = os.getenv('base_mysql_pwd')
mysql_db = os.getenv('app_fresh_db')

mysql_config = {
    'host': mysql_host,
    'port': mysql_port,
    'user': mysql_user,
    'password': mysql_pwd,
    'database': mysql_db
}

# 读取环境变量 - 目标feed
fresh_entry_table = os.getenv('app_fresh_entry_table')
fresh_feed_table = os.getenv('app_fresh_feed_table')
target_feed_list = os.getenv('add_feed_target').split(',')
target_feed_list = [int(item) for item in target_feed_list]

# 查询目标数据
in_clause = ','.join(['%s'] * len(target_feed_list))
feed_filter = ', '.join([f'"{item}"' for item in target_feed_list])

entry_query_sql = f'''
SELECT id, title, author, UNCOMPRESS(content_bin) AS content, 
link, date, lastSeen, hash, 
is_read, is_favorite, id_feed, tags, attributes
FROM {fresh_entry_table} WHERE id_feed IN ({feed_filter})
AND date >= %s AND date <= %s
'''

entry_data = get_entry_list(mysql_config, entry_query_sql, 'hourly')

if entry_data:

    format_list = []

    for each in entry_data:

        update_dict = {
            'feed': f"feed_{each['id']}",
            'title': each['title'],
            'content': each['content'].decode('utf-8'),
            'img': '',
            'link': each['feed_link'],
            'dt': each['date'],
            'resource': str(each['id_feed'])
        }

        soup = BeautifulSoup(update_dict['feed_content'],'html.parser')

        try:     
            if soup.find():
                images = soup.find_all('img')
                src_list = [img.get('src') for img in images if img.get('src')]
                update_dict['img'] = src_list[0]

            else:
                update_dict['img'] = None

        except Exception as e:
            update_dict['img'] = None

        format_list.append(update_dict)

    feed_url = 'http://env_wxfeeds:8000/feed/feedAdd'
    feed_data = {'data':format_list}
    feed_rslt = requests.post(feed_url, json=feed_data)

    print(feed_rslt)

else:
    print('no feed data needed tobe add')