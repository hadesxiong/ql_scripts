# coding=utf8
import requests,os,pymysql
from bson import ObjectId
from datetime import datetime

# 读取环境变量
app_wf_apikey = os.getenv('app_wf_api_key')
city_list = os.getenv('app_wf_normal_city').split(',')
agd_weahterApi_url = os.getenv('app_wf_api_url')

# 链接数据库
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

wf_base_table = os.getenv('app_wf_table_base')
insert_sql = f'''
INSERT INTO {wf_base_table} (
    province, city, adcode, report_time, request_id,
    cast_type, cast_date, date_week, weather, temp,
    temp_float, wind, power, humidity, humidity_float
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s
)
'''

rslt_list = []

# 遍历常规城市
for each_city in city_list:

    wf_params = {
        'key': app_wf_apikey,
        'city': each_city,
        'extensions': 'base',
        'output': 'JSON'
    }

    wf_rslt = requests.get(agd_weahterApi_url,wf_params)

    if wf_rslt.status_code == 200 and wf_rslt.json().get('infocode',None) == '10000':

        try:
            wf_data = wf_rslt.json()['lives'][0]
            request_id = f'wfreq_{ObjectId()}'

            for each in wf_data:

                date_string = each['reporttime'][0:9]
                date_object = datetime.strptime(date_string, "%Y-%m-%d")
                day_of_week = date_object.weekday() + 1

                rslt_list.append({
                    'province': each['province'],
                    'city': each['city'],
                    'adcode': each['adcode'],
                    'report_time': each['reporttime'],
                    'request_id': request_id,
                    'cast_type': 'base',
                    'cast_date': each['reporttime'][0:9],
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

try:
    with mysql_connection.cursor() as cursor:
        value_list = [{
            data['province'], data['city'], data['adcode'],
            data['report_time'], data['request_id'], data['cast_type'],
            data['cast_date'], data['date_week'], data['weather'],
            data['temp'], data['temp_float'], data['wind'], data['power'],
            data['humidity'], data['humidity_float']
        } for data in rslt_list]

        cursor.executemany(insert_sql,value_list)
        mysql_connection.commit()

except Exception as e:
    print(e)

if mysql_connection:
    mysql_connection.close()