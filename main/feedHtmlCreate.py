# coding=utf8
import os,sys
from string import Template
from datetime import datetime
# 添加引用路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from app_freshrss.feedParse import get_entry_list
from app_feeds.genHtml import gen_html, format_entry_list
from app_feeds.convertPic import get_page_height, convert_html_png

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

container_template = '''
    <div class="container">
        <div class="title_con">
            <div class="line_con"><div class="title_line"></div></div>
            <div class="title_main">${title_main}</div>
        </div>
        ${ban_data}
        <div class="feed_con">
        ${ban_before_data}
            <div class="feed_abst">
                ${feed_abst}
            </div>
        ${ban_after_data}
        </div>
        <hr class="split_line">
    </div>
'''

with open('/ql/data/scripts/utils/app_feeds/template/tmpl_tech.html','r',encoding='utf-8') as html_file:
    html_template = Template(html_file.read())

    target_file = gen_html(
        html_tmpl=html_template,con_tmpl=container_template,
        feeds_data=entry_list, target_dir='/ql/data/scripts/output/feeds'
    )

    chrome_driver_path = '/usr/bin/chromedriver'

    page_height = get_page_height(target_file,chrome_driver_path)

    if page_height:
        
        today_dt = datetime.now().strftime('%Y_%m_%d')

        convert_html_png(
            html_file=target_file, 
            css_file='/ql/data/scripts/utils/app_feeds/template/tmpl_tech.css',
            ot_path='/ql/data/scripts/output/img_feeds/', save_as=f'{today_dt}_tech_feeds.png',
            img_width=824, img_height=int(page_height)
        )

    else:

        print('height not found, cannot convert html files to png')