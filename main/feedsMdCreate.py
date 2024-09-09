# coding=utf8
import os,sys
# 添加引用路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from datetime import datetime, timedelta
# from app_freshrss.feedsParse import feeds_parse
from base_db.baseMysql import start_custom_connection
from base_db.baseMysql import close_connection
from base_db.baseMysql import write_many, fetch_all

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

today_midnight = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
tomorrow_midnight = today_midnight + timedelta(days=1)

start_ts = int(today_midnight.timestamp())
end_ts = int(tomorrow_midnight.timestamp()) - 1

entry_query_sql = f'''
SELECT id, title, author, UNCOMPRESS(content_bin) AS content, 
link, date, lastSeen, hash, 
is_read, is_favorite, id_feed, tags, attributes
FROM {fresh_entry_table} WHERE id_feed IN ({feed_filter})
AND is_read = 0
AND date >= %s AND date <= %s
'''

mysql_connection = start_custom_connection(mysql_config)
results = fetch_all(mysql_connection, entry_query_sql, (start_ts, end_ts))

for each in results:
    print(each['content'].decode('utf-8'))

close_connection(mysql_connection)

