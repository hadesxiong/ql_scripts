# coding=utf8
import json,os,sys
# 添加引用路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from app_wf.getCast import get_cast
from app_nfyserver.userLogin import user_login
from app_nfyserver.notifyPush import notify_push
from base_db.baseMysql import start_connection, close_connection, write_many

# 读取城市列表及数据表名
app_wf_city_list = os.getenv('app_wf_city_list').split(',')
app_wf_table_all = os.getenv('app_wf_table_all')
app_wf_table_base = os.getenv('app_wf_table_base')

# 获取预报数据
wf_data_all = get_cast(app_wf_city_list,'all')
wf_data_base = get_cast(app_wf_city_list,'base')

wf_data_all_sql = wf_data_all['sql_data']
wf_data_base_sql = wf_data_base['sql_data']

wf_flat_all = wf_data_all['flat'][0]
wf_flat_base = wf_data_base['flat'][0]

# 读取推送功能用户名
app_nfy_user = os.getenv('app_nfy_user')
app_nfy_pwd = os.getenv('app_nfy_pwd')

# 获取推送频道信息
app_nfy_chnl_bark = f'chnl_{os.getenv("app_nfy_chnl_barkDaily")}'
app_nfy_chnl_ntfy = f'chnl_{os.getenv("app_nfy_chnl_ntfyDaily")}'

# 获取推送模版信息
app_nfy_tmpl_barkWF = f'tmpl_{os.getenv("app_nfy_tmpl_barkWF")}'
app_nfy_tmpl_ntfyWF = f'tmpl_{os.getenv("app_nfy_tmpl_ntfyWF")}'

# 获取推送目标
rcv_bark_wf = json.loads(os.getenv('rcv_bark_wf'))
rcv_ntfy_wf = json.loads(os.getenv('rcv_ntfy_wf'))


app_nfy_token = user_login(app_nfy_user,app_nfy_pwd)

wf_nfy_data = {
    'today_date': wf_flat_all['cast_date'],
    'sh_sh_dwth': wf_flat_all['dayweather'],
    'sh_sh_nwth': wf_flat_all['nightweather'],
    'sh_sh_dtemp': wf_flat_all['daytemp'],
    'sh_sh_ntemp': wf_flat_all['nighttemp'],
    'sh_sh_wdd': wf_flat_base['wind'],
    'sh_sh_wdp': wf_flat_base['power'],
    'sh_sh_hmd': wf_flat_base['humidity']
}

# 推送至bark
bark_push_res = notify_push(
    token=app_nfy_token, channel=app_nfy_chnl_bark,
    template=app_nfy_tmpl_barkWF, tmpl_type=1,
    rcv_type=rcv_bark_wf['type'],
    rcv_target = rcv_bark_wf['target'],
    msg_dict=wf_nfy_data, call_from='ql_scripts_daily'
)

print(bark_push_res)

# 推送至ntfy
ntfy_push_res = notify_push(
    token=app_nfy_token, channel=app_nfy_chnl_ntfy,
    template=app_nfy_tmpl_ntfyWF, tmpl_type=2,
    rcv_type=rcv_ntfy_wf['type'],
    rcv_target=rcv_ntfy_wf['target'],
    msg_dict=wf_nfy_data, call_from='ql_scripts_daily'
)

print(ntfy_push_res)

# 数据库存储
mysql_con = start_connection()

wf_all_sql = f"""
INSERT INTO {app_wf_table_all} (
    province, city, adcode, report_time, request_id, 
    cast_type, cast_date, date_week, dayweather, nightweather, 
    daytemp, nighttemp, daytemp_float, nighttemp_float, daywind, 
    daypower, nightwind, nightpower
) VALUES (
    %s, %s, %s, %s, %s, 
    %s, %s, %s, %s, %s, 
    %s, %s, %s, %s, %s,
    %s, %s, %s
)
"""

wf_base_sql = f'''
INSERT INTO {app_wf_table_base} (
    province, city, adcode, report_time, request_id,
    cast_type, cast_date, date_week, weather, temp,
    temp_float, wind, power, humidity, humidity_float
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s
)
'''

write_many(mysql_con, wf_all_sql, wf_data_all_sql)
write_many(mysql_con, wf_base_sql, wf_data_base_sql)

close_connection(mysql_con)