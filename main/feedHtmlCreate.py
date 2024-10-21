# coding=utf8
import os,sys,glob
from string import Template
from datetime import datetime
# 添加引用路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from app_freshrss.feedParse import get_entry_list
from app_feeds.genHtml import gen_html, format_entry_list
from app_feeds.convertPic import get_page_height, convert_html_png
from base_wechat.baseApi import get_wx_token, upload_wx_material, create_md_draft

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
target_feed_list = os.getenv('app_feeds_target').split(',')
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

# 上传到微信
wx_appid = os.getenv('wx_official_appid_dxmz')
wx_secret = os.getenv('wx_official_secret_dxmz')

wx_token = get_wx_token(wx_appid,wx_secret)

# 判断html的图片以及头图是否正常生成
main_img_filePath = f'/ql/data/scripts/output/img_feeds/{today_dt}_tech_feeds.png'
main_img_fileName = f'{today_dt}_tech_feeds.png'
main_img_exist = os.path.exists(main_img_filePath)

head_img_pattern = f'/ql/data/scripts/output/feeds/img/{today_dt}_*.*'

head_img_fileList = sorted(glob.glob(head_img_pattern),key=os.path.basename)

head_img_exist = len(head_img_fileList) > 0

if main_img_exist and head_img_exist and wx_token.get('token',None):

    # 上传头图
    main_upload_result = upload_wx_material(token=wx_token['token'], file_type='image', 
                                            file_name=main_img_fileName, file_path=main_img_filePath)
    
    head_img_filePath = head_img_fileList[0]
    head_img_fileName = head_img_filePath.split('/')[-1]

    head_upload_result = upload_wx_material(token=wx_token['token'], file_type='image',
                                            file_name=head_img_fileName, file_path=head_img_filePath)
    
    print(main_upload_result,head_upload_result)

else:
    main_upload_result, head_upload_result = {'result':False},{'result':False}

if main_upload_result.get('media_id',None) and head_upload_result.get('media_id',None):

    date_str = datetime.now().strftime('%Y.%m.%d')
    article_data = {
        'title': f'【{date_str}】Tech新闻摘要',
        'author': "Nero's BOT",
        'digest': '',
        'content': f'<img src={main_upload_result["url"]}>',
        'thumb_media_id': head_upload_result['media_id']
    }

    article_list = [article_data]

    draft_result = create_md_draft(token=wx_token['token'],article_list=article_list)

    print(draft_result)

else:

    print('upload_failed')