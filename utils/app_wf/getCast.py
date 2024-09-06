# coding=utf8
import os,requests

from bson import ObjectId
from datetime import datetime

# 读取环境变量
app_wf_api_key = os.getenv('app_wf_api_key')
app_wf_api_url = os.getenv('app_wf_api_url')

def get_cast(city_list: list, type:str) -> list:

    rslt_list,flat_list = [],[]
    
    wf_params = {
        'key': app_wf_api_key,
        'city': '',
        'extensions': type,
        'output': 'JSON'
    }

    for each_city in city_list:

        wf_params['city'] = each_city
        each_rslt = requests.get(app_wf_api_url,wf_params)

        if each_rslt.status_code == 200 and  \
            each_rslt.json().get('infocode',None) == '10000':
            
            rslt_list.append(each_rslt)

    if type == 'all':

        for each in rslt_list:

            common_info = each.json()['forecasts'][0]
            detail_info = each.json()['forecasts'][0]['casts']

            request_id = f'wfreq_{ObjectId()}'

            for each_d in detail_info:
                
                flat_list.append({
                    'province': common_info['province'],
                    'city': common_info['city'],
                    'adcode': common_info['adcode'],
                    'report_time': common_info['reporttime'],
                    'request_id': request_id,
                    'cast_type':'all',
                    'cast_date': each_d['date'],
                    'date_week': int(each_d['week']),
                    'dayweather': each_d['dayweather'],
                    'nightweather': each_d['nightweather'],
                    'daytemp': int(each_d['daytemp']),
                    'nighttemp': int(each_d['nighttemp']),
                    'daytemp_float': float(each_d['daytemp_float']),
                    'nighttemp_float': float(each_d['nighttemp_float']),
                    'daywind': each_d['daywind'],
                    'daypower': each_d['daypower'],
                    'nightwind':each_d['nightwind'],
                    'nightpower': each_d['nightpower']
                })

        sql_data =  [(
            data["province"], data["city"], data["adcode"],
            data["report_time"], data["request_id"],
            data["cast_type"], data["cast_date"],
            data["date_week"], data["dayweather"],
            data["nightweather"], data["daytemp"],
            data["nighttemp"], data["daytemp_float"],
            data["nighttemp_float"], data["daywind"],
            data["daypower"], data["nightwind"],
            data["nightpower"]
        ) for data in flat_list]
    
    elif type == 'base':

        for each in rslt_list:

            detail_info = each.json()['lives']
            request_id = f'wfreq_{ObjectId()}'

            for each_d in detail_info:

                date_string = each_d['reporttime'][0:10]
                date_object = datetime.strptime(date_string, "%Y-%m-%d")
                day_of_week = date_object.weekday() + 1

                flat_list.append({
                    'province': each_d['province'],
                    'city': each_d['city'],
                    'adcode': each_d['adcode'],
                    'report_time': each_d['reporttime'],
                    'request_id': request_id,
                    'cast_type': 'base',
                    'cast_date': each_d['reporttime'][0:10],
                    'date_week': day_of_week,
                    'weather': each_d['weather'],
                    'temp': each_d['temperature'],
                    'temp_float': each_d['temperature_float'],
                    'wind': each_d['winddirection'],
                    'power': each_d['windpower'],
                    'humidity': each_d['humidity'],
                    'humidity_float': each_d['humidity_float']
                })

        sql_data = [(
            data['province'], data['city'], data['adcode'],
            data['report_time'], data['request_id'], data['cast_type'],
            data['cast_date'], data['date_week'], data['weather'],
            data['temp'], data['temp_float'], data['wind'], data['power'],
            data['humidity'], data['humidity_float']
        ) for data in flat_list]

    return {'flat': flat_list,'sql_data':sql_data}