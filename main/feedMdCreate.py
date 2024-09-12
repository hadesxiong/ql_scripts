# coding=utf8
import os,sys
# 添加引用路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from app_freshrss.feedParse import get_entry_list,format_entry_list
from app_freshrss.feedParse import create_feed_md

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
target_feed_list = os.getenv('app_fresh_target').split(',')
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

entry_data = get_entry_list(mysql_config, entry_query_sql, 'day')
entry_list = format_entry_list(entry_data)

entry_md = create_feed_md(entry_list)
print(entry_md)