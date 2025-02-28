# coding=utf8
import json,os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from datetime import datetime,timedelta,time

# 引入方法
from app_tb.tbApi import getToken,createTask

# 登陆获取token
access_token = getToken()

# 获取每日任务
app_tb_dailyPlan = os.getenv('app_tb_dailyPlan')
app_tb_gameProj = os.getenv('app_tb_gameProj')
app_tb_creator = os.getenv('app_tb_creator')

game_todo_dict = json.loads(app_tb_dailyPlan)['game']

now_time = datetime.now()
today_date = now_time.date()
today_str = today_date.strftime('%Y-%m-%d')

for each in game_todo_dict:

    if (each['type']) == 'daily':
        
        # 创建父任务
        parent_data = {
            'projectId': app_tb_gameProj,
            'content': f'{each["title"]}-{today_str}',
            'executorId': app_tb_creator,
            'startDate': f'{today_str} 07:00:00',
            'dueDate': f'{today_str} 23:00:00',
            'note': each['note'],
            'tagIds':each['tagIds']
        }
        parent_res = createTask(token=access_token, data=parent_data)

        if len(each['items']) > 0:

            parent_id = parent_res['result']['taskId']

            for each_sub in each['items']:

                sub_data = {
                    'projectId': app_tb_gameProj,
                    'content': f'{each_sub["title"]}-{today_str}',
                    'executorId': app_tb_creator,
                    'startDate': f'{today_str} 07:00:00',
                    'dueDate': f'{today_str} 23:00:00',
                    'note': each_sub['note'],
                    'tagIds':each_sub['tagIds'],
                    'parentTaskId': parent_id
                }

                sub_res = createTask(token=access_token, data=sub_data)

    elif (each['type']) == 'Thur' and today_date.weekday() == 3:

        # 创建父任务
        parent_data = {
            'projectId': app_tb_gameProj,
            'content': f'{each["title"]}-{today_str}',
            'executorId': app_tb_creator,
            'startDate': (datetime.combine(today_date,time(7,0)) - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M'),
            'dueDate': (datetime.combine(today_date,time(23,0)) - timedelta(hours=8) + timedelta(days=6)).strftime('%Y-%m-%d %H:%M'),
            'note': each['note'],
            'tagIds': each['tagIds'],
        }
        parent_res = createTask(token=access_token, data=parent_data)

        if len(each['items']) > 0:

            parent_id = parent_res['result']['taskId']

            for each_sub in each['items']: 

                sub_data = {
                    'projectId': app_tb_gameProj,
                    'content': f'{each_sub["title"]}-{today_str}',
                    'executorId': app_tb_creator,
                    'startDate': (datetime.combine(today_date,time(7,0)) - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M'),
                    'dueDate': (datetime.combine(today_date,time(23,0)) - timedelta(hours=8) + timedelta(days=6)).strftime('%Y-%m-%d %H:%M'),
                    'note': each_sub['note'],
                    'tagIds': each_sub['tagIds'],
                    'parentTaskId': parent_id
                }

                sub_res = createTask(token=access_token, data=sub_data)

print('创建成功')