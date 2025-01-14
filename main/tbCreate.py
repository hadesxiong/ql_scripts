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

game_todo_dict = json.loads(app_tb_dailyPlan)['游戏']

now_time = datetime.now()
today_date = now_time.date()
today_str = today_date.strftime('%Y-%m-%d')

for each in game_todo_dict:

    if (each['type']) == 'daily':
        
        # 创建父任务
        parent_data = {
            'projectId': app_tb_gameProj,
            'content': f'{each["title"]}-{today_str}',
            'startDate': f'{today_str} 07:00:00',
            'dueDate': f'{today_str} 23:00:00',
            'note': '系统脚本定时创建任务',
            'tagIds':[]
        }
        parent_res = createTask(token=access_token, data=parent_data)

        parent_id = parent_res['result']['taskId']

        for each_sub in each['items']:

            sub_data = {
                'projectId': app_tb_gameProj,
                'content': f'{each_sub}-{today_str}',
                'startDate': f'{today_str} 07:00:00',
                'dueDate': f'{today_str} 23:00:00',
                'note': '系统脚本定时创建任务',
                'tagIds':[],
                'parentTaskId': parent_id
            }

            sub_res = createTask(token=access_token, data=sub_data)

    elif (each['type']) == 'Thur' and today_date.weekday() == 3:

        # 创建父任务
        parent_data = {
            'projectId': app_tb_gameProj,
            'content': f'{each["title"]}-{today_str}',
            'startDate': f'{today_str} 07:00:00',
            'dueDate'
        }