# coding=utf8
import base64,os,pymysql,requests
from bson import ObjectId
from datetime import datetime
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes


# 读取环境变量
app_wf_api_key = os.getenv('app_wf_api_key')
app_wf_api_url = os.getenv('app_wf_api_url')
app_wf_all_city = os.getenv('app_wf_all_city').split(',')
app_wf_base_city = os.getenv('app_wf_base_city').split(',')

# 配置数据库
mysql_host = os.getenv('base_mysql_host')
mysql_port = int(os.getenv('base_mysql_port'))
mysql_user = os.getenv('base_mysql_user')
mysql_pwd = os.getenv('base_mysql_pwd')
mysql_db = os.getenv('base_qinglong_db')

mysql_config = {
    'host': mysql_host,
    'port': mysql_port,
    'user': mysql_user,
    'password': mysql_pwd,
    'database': mysql_db
}

mysql_connection = pymysql.connect(**mysql_config)

wf_table_all = os.getenv('app_wf_table_all')
wf_table_base = os.getenv('app_wf_table_base')

wf_all_sql = f"""
INSERT INTO {wf_table_all} (
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
INSERT INTO {wf_table_base} (
    province, city, adcode, report_time, request_id,
    cast_type, cast_date, date_week, weather, temp,
    temp_float, wind, power, humidity, humidity_float
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s
)
'''

# 预报
wf_all_rslt_list = []

for each_city in app_wf_all_city:

    wf_all_params = {
        'key': app_wf_api_key,
        'city': each_city,
        'extensions': 'all',
        'output': 'JSON'
    }

    wf_all_rslt = requests.get(app_wf_api_url,wf_all_params)

    if wf_all_rslt.status_code == 200 and wf_all_rslt.json().get('infocode',None) == '10000':

        try:
            wf_common = wf_all_rslt.json()['forecasts'][0]
            wf_detail = wf_all_rslt.json()['forecasts'][0]['casts']

            request_id = f'wfreq_{ObjectId()}'

            for each in wf_detail:
                wf_all_rslt_list.append({
                    'province': wf_common['province'],
                    'city': wf_common['city'],
                    'adcode': wf_common['adcode'],
                    'report_time': wf_common['reporttime'],
                    'request_id': request_id,
                    'cast_type':'all',
                    'cast_date': each['date'],
                    'date_week': int(each['week']),
                    'dayweather': each['dayweather'],
                    'nightweather': each['nightweather'],
                    'daytemp': int(each['daytemp']),
                    'nighttemp': int(each['nighttemp']),
                    'daytemp_float': float(each['daytemp_float']),
                    'nighttemp_float': float(each['nighttemp_float']),
                    'daywind': each['daywind'],
                    'daypower': each['daypower'],
                    'nightwind':each['nightwind'],
                    'nightpower': each['nightpower']
                })

        except Exception as e:
            print(e)

# try:
#     with mysql_connection.cursor() as cursor:
#         value_list = [(
#             data["province"], data["city"], data["adcode"],
#             data["report_time"], data["request_id"],
#             data["cast_type"], data["cast_date"],
#             data["date_week"], data["dayweather"],
#             data["nightweather"], data["daytemp"],
#             data["nighttemp"], data["daytemp_float"],
#             data["nighttemp_float"], data["daywind"],
#             data["daypower"], data["nightwind"],
#             data["nightpower"]
#         ) for data in wf_all_rslt_list]

#         cursor.executemany(wf_all_sql,value_list)
#         mysql_connection.commit()

# except Exception as e:
#     print(e)

# 实时
wf_base_rslt_list = []

for each_city in app_wf_base_city:

    wf_base_params = {
        'key': app_wf_api_key,
        'city': each_city,
        'extensions': 'base',
        'output': 'JSON'
    }

    wf_base_rslt = requests.get(app_wf_api_url,wf_base_params)

    if wf_base_rslt.status_code == 200 and wf_base_rslt.json().get('infocode',None) == '10000':

        try:
            wf_data = wf_base_rslt.json()['lives']
            request_id = f'wfreq_{ObjectId()}'

            for each in wf_data:

                date_string = each['reporttime'][0:10]
                date_object = datetime.strptime(date_string, "%Y-%m-%d")
                day_of_week = date_object.weekday() + 1

                wf_base_rslt_list.append({
                    'province': each['province'],
                    'city': each['city'],
                    'adcode': each['adcode'],
                    'report_time': each['reporttime'],
                    'request_id': request_id,
                    'cast_type': 'base',
                    'cast_date': each['reporttime'][0:10],
                    'date_week': day_of_week,
                    'weather': each['weather'],
                    'temp': each['temperature'],
                    'temp_float': each['temperature_float'],
                    'wind': each['winddirection'],
                    'power': each['windpower'],
                    'humidity': each['humidity'],
                    'humidity_float': each['humidity_float']
                })

        except Exception as e:
            print(str(e))

# try:
#     with mysql_connection.cursor() as cursor:
#         value_list = [{
#             data['province'], data['city'], data['adcode'],
#             data['report_time'], data['request_id'], data['cast_type'],
#             data['cast_date'], data['date_week'], data['weather'],
#             data['temp'], data['temp_float'], data['wind'], data['power'],
#             data['humidity'], data['humidity_float']
#         } for data in wf_base_rslt_list]

#         cursor.executemany(wf_base_sql,value_list)
#         mysql_connection.commit()

# except Exception as e:
#     print(e)

sh_cast_all, sh_cast_base = wf_all_rslt_list[0], wf_base_rslt_list[0]


# 拼接推送消息
cast_msg_data = {
    "today_date": sh_cast_all['cast_date'],
    "sh_sh_dwth": sh_cast_all['dayweather'],
    "sh_sh_nwth": sh_cast_all['nightweather'],
    "sh_sh_dtemp": sh_cast_all['daytemp'],
    "sh_sh_ntemp": sh_cast_all['nighttemp'],
    "sh_sh_wdd": sh_cast_base['wind'],
    "sh_sh_wdp": sh_cast_base['power'],
    "sh_sh_hmd": sh_cast_base['humidity']
}

# 读取消息推送配置
app_nfy_user = os.getenv('app_nfy_user')
app_nfy_pwd = os.getenv('app_nfy_pwd')
app_nfy_key = os.getenv('app_nfy_key')
app_nfy_iv = os.ge