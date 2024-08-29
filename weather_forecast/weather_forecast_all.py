# coding=utf8
import requests,os,pymysql
from bson import ObjectId

# 读取环境变量
agd_weatherApi_key = os.getenv('wf_agd_weatherApi_key')
city_list = os.getenv('wf_city_list').split(',')
agd_weahterApi_url = os.getenv('wf_agd_weatherApi_url')

# 链接数据库
mysql_host = os.getenv('mysql_sit1_host')
mysql_port = int(os.getenv('mysql_sit1_port'))
mysql_user = os.getenv('mysql_sit1_user')
mysql_pwd = os.getenv('mysql_sit1_pwd')
mysql_db = os.getenv('qinglong_db')

mysql_config = {
    'host': mysql_host,
    'port': mysql_port,
    'user': mysql_user,
    'password': mysql_pwd,
    'database': mysql_db
}

mysql_connection = pymysql.connect(**mysql_config)

wf_table_name = os.getenv('wf_table_name')
insert_sql = f"""
INSERT INTO {wf_table_name} (
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

rslt_list = []

# 遍历城市
for each_city in city_list:

    wf_params = {
        'key': agd_weatherApi_key,
        'city': each_city,
        'extensions': 'all',
        'output': 'JSON'
    }

    wf_rslt = requests.get(agd_weahterApi_url,wf_params)

    if wf_rslt.status_code == 200 and wf_rslt.json().get('infocode',None) == '10000':

        try:
            wf_common = wf_rslt.json()['forecasts'][0]
            wf_detail = wf_rslt.json()['forecasts'][0]['casts']

            request_id = f'wfreq_{ObjectId()}'

            for each in wf_detail:
                rslt_list.append({
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

try:
    with mysql_connection.cursor() as cursor:
        value_list = [(
            data["province"], data["city"], data["adcode"],
            data["report_time"], data["request_id"],
            data["cast_type"], data["cast_date"],
            data["date_week"], data["dayweather"],
            data["nightweather"], data["daytemp"],
            data["nighttemp"], data["daytemp_float"],
            data["nighttemp_float"], data["daywind"],
            data["daypower"], data["nightwind"],
            data["nightpower"]
        ) for data in rslt_list]

        cursor.executemany(insert_sql,value_list)
        mysql_connection.commit()

except Exception as e:
    print(e)

if mysql_connection: 
    mysql_connection.close()