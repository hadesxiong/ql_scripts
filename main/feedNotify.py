# coding=utf8
import json,os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from app_freshrss.feedParse import get_entry_list,format_entry_list
from app_freshrss.feedParse import create_feed_msg, set_feed_read
from app_nfyserver.userLogin import user_login
from app_nfyserver.notifyPush import notify_push

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
AND is_read = 0
AND date >= %s AND date <= %s
'''

entry_update_sql = f'''
UPDATE {fresh_entry_table} SET is_read = 1 WHERE id = %s
'''

# 读取推送功能用户名
app_nfy_user = os.getenv('app_nfy_user')
app_nfy_pwd = os.getenv('app_nfy_pwd')

# 获取推送频道信息
app_nfy_chnl_bark = f'chnl_{os.getenv("app_nfy_chnl_barkDaily")}'
app_nfy_chnl_ntfy = f'chnl_{os.getenv("app_nfy_chnl_ntfyDaily")}'

# 获取推送模版信息
app_nfy_tmpl_barkXW = f'tmpl_{os.getenv("app_nfy_tmpl_barkXW")}'
app_nfy_tmpl_ntfyXW = f'tmpl_{os.getenv("app_nfy_tmpl_ntfyXW")}'

# 获取推送目标
rcv_bark_feeds = json.loads(os.getenv('rcv_bark_feeds'))
rcv_ntfy_feeds = json.loads(os.getenv('rcv_ntfy_feeds'))


entry_data = get_entry_list(mysql_config,entry_query_sql,'timely')
if entry_data:
    entry_list = format_entry_list([entry_data])
    entry_dict = create_feed_msg(entry_list[0])

    app_nfy_token = user_login(app_nfy_user,app_nfy_pwd)
    news_data = entry_dict['data']
    news_url = {'feed_url': entry_dict['url']}

    # 推送至bark
    bark_push_res = notify_push(
        token=app_nfy_token, channel=app_nfy_chnl_bark,
        template=app_nfy_tmpl_barkXW, tmpl_type=1,
        rcv_type=rcv_bark_feeds['type'],
        rcv_target=rcv_bark_feeds['target'],
        msg_dict=news_data, call_from='ql_scripts_daily',
        args=news_url
    )

    print(bark_push_res)

    # 推送至ntfy
    ntfy_push_res = notify_push(
        token=app_nfy_token, channel=app_nfy_chnl_ntfy,
        template=app_nfy_tmpl_ntfyXW, tmpl_type=2,
        rcv_type=rcv_ntfy_feeds['type'],
        rcv_target=rcv_ntfy_feeds['target'],
        msg_dict=news_data, call_from='ql_scripts_daily',
        args=news_url
    )

    print(ntfy_push_res)

    # 更新数据
    set_feed_read(mysql_config,fresh_entry_table,[(entry_data['id'])])


else:
    print('no new entry found.')