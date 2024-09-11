# coding=utf8
import html,os,sys
# 添加引用路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils')))

from bs4 import BeautifulSoup

from base_db.baseMysql import start_custom_connection
from base_db.baseMysql import fetch_all, write_many
from base_db.baseMysql import close_connection

from datetime import datetime, timedelta

def get_entry_list(mysql_config:dict, sql:str, type:str):

    mysql_ins = start_custom_connection(mysql_config)

    if type == 'day':

        today_midnight = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
        tomorrow_midnight = today_midnight + timedelta(days=1)

        start_ts = int(today_midnight.timestamp())
        end_ts = int(tomorrow_midnight.timestamp()) - 1

        results = fetch_all(mysql_ins, sql, (start_ts,end_ts))

    close_connection(mysql_ins)

    return results

def format_entry_list(entry_list: list):

    format_list = []

    for each in entry_list:
        
        entry_content = each['content'].decode('utf-8')
        entry_dict = {
            'entry_id': f"entry_{each['id']}",
            'id_feed': each['id_feed'],
            'title': each['title'],
            'img': '',
            'content': entry_content,
            'link': each['link'],
            'date': each['date'],
            'tags': each['tags']
        }

        soup = BeautifulSoup(entry_content,'html.parser')
        
        try:     
            if soup.find():
                images = soup.find_all('img')
                src_list = [img.get('src') for img in images if img.get('src')]
                entry_dict['img'] = src_list[0]

            else:
                entry_dict['img'] = 'blank'

        except Exception as e:
            entry_dict['img'] = 'blank'

        format_list.append(entry_dict)

    return format_list