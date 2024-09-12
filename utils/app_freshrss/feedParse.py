# coding=utf8
import jieba.analyse,os,requests,sys
# 添加引用路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils')))

from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from base_db.baseMysql import start_custom_connection
from base_db.baseMysql import fetch_all, write_many, fetch_one
from base_db.baseMysql import close_connection

# 获取当天的所需feed
def get_entry_list(mysql_config:dict, sql:str, type:str):

    mysql_ins = start_custom_connection(mysql_config)

    # 获取今日的数据
    if type == 'day':

        # 昨天晚上21点到今天21点
        today_midnight = datetime.now().replace(hour=21,minute=0,second=0,microsecond=0) - timedelta(days=1)
        tomorrow_midnight = today_midnight + timedelta(days=1)

        start_ts = int(today_midnight.timestamp())
        end_ts = int(tomorrow_midnight.timestamp()) - 1

        results = fetch_all(mysql_ins, sql, (start_ts,end_ts))

    elif type == 'timely':

        # today_midnight = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
        # tomorrow_midnight = today_midnight + timedelta(days=1)
        today_midnight = datetime.now().replace(hour=21,minute=0,second=0,microsecond=0) - timedelta(days=1)
        tomorrow_midnight = today_midnight + timedelta(days=1)

        start_ts = int(today_midnight.timestamp())
        end_ts = int(tomorrow_midnight.timestamp()) - 1

        results = fetch_one(mysql_ins, sql, (start_ts,end_ts))

    close_connection(mysql_ins)
    return results

# 格式化entry内容,形成数组[字典]
def format_entry_list(entry_list: list):

    format_list = []

    # 读取来源映射字典
    feed_dict = os.getenv('app_fresh_target_dict')

    for each in entry_list:
        
        entry_content = each['content'].decode('utf-8')
        entry_dict = {
            'entry_id': f"entry_{each['id']}",
            'id_feed': feed_dict[str(each['id_feed'])],
            'title': each['title'],
            'img': '',
            'content': entry_content,
            'link': each['link'],
            'date': each['date'],
        }

        soup = BeautifulSoup(entry_content,'html.parser')

        content_text = soup.get_text(strip=True)
        entry_dict['summary'] = f'{content_text[0:100]}...'
        entry_dict['tags'] = jieba.analyse.extract_tags(content_text,topK=3)
        
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

# 生成标准化的markdown文件
def create_feed_md(entry_list:list) -> str:

    today = datetime.now()

    # 固定部分补充
    markdown_content = f"#### Feeds推送 - {today.year}年{today.month:02d}月{today.day:02d}日 \n\n"
    
    for each in entry_list:

        markdown_content += f"#### {each['title']}\n"
        markdown_content += f"<img src='{each['img']}' width='600'>\n"
        markdown_content += f"##### 来源: {str(each['id_feed'])}\n"
        markdown_content += f"{each['summary']}\n"
        markdown_content += f"[查看原文]({each['link']})\n"

        tags_str = ','.join([f'#{item}' for item in each['tags']])
        markdown_content += f'关键词：{tags_str}\n\n'

    return markdown_content

def create_feed_msg(entry_data:dict) -> dict:

    feed_data = {
        "feed_title": entry_data['title'],
        "feed_abst": entry_data['summary'],
        "feed_origin": entry_data['id_feed']
    }

    feed_url = entry_data['link']

    return {'data':feed_data,'url':feed_url}

def set_feed_read(mysql_config:dict, table_name:str, data_list:list):

    mysql_ins = start_custom_connection(mysql_config)

    update_sql = f'''
    UPDATE {table_name} SET is_read = 1 WHERE id = %s
    '''

    print(update_sql)
    print(data_list)

    write_many(mysql_ins,update_sql,data_list)

    close_connection(mysql_ins)


