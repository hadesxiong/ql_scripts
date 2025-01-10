# coding=utf8
import json,os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from datetime import datetime,timedelta,time

# 引入方法
from app_wekan.wekanApi import userLogin, getSwimLanes, getLists, createCard, editCard, createCheckList

app_wekan_user = os.getenv('app_wekan_user')
app_wekan_pwd = os.getenv('app_wekan_pwd')
app_wekan_board = os.getenv('app_wekan_board')
app_wekan_author = os.getenv('app_wekan_author')

# 登陆获取token
token_dict = userLogin(app_wekan_user, app_wekan_pwd)
token_str = token_dict['token']

# 获取泳道 - 进行中
swimlane_list = getSwimLanes(token_str, app_wekan_board)
work_swimlane = next((item for item in swimlane_list if item['title'] == '工作'), None)['_id']
game_swimlane = next((item for item in swimlane_list if item['title'] == '游戏'), None)['_id']
learn_swimlane = next((item for item in swimlane_list if item['title'] == '学习'), None)['_id']

# 获取列表
wekan_list = getLists(token_str, app_wekan_board)
doing_list = next((item for item in wekan_list if item['title'] == '处理中'), None)['_id']

# 获取每日任务
app_wekan_dailyPlan = os.getenv('app_wekan_dailyPlan')

game_todo_dict = json.loads(app_wekan_dailyPlan)['游戏']

now_time = datetime.now()
adjust_time = now_time - timedelta(hours=8)

today_date = now_time.date()
today_str = today_date.strftime('%Y-%m-%d')

# 创建任务卡
for each in game_todo_dict:

    if each['type'] == "daily":
        card_res = createCard(
            token=token_str, board=app_wekan_board, list=doing_list, 
            author=app_wekan_author, title=f"{each['title']}-{today_str}",
            desc=f"{each['desc']}",swimlane=game_swimlane)
        
        edit_data = {
            "startAt": (datetime.combine(today_date,time(7,0)) - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M'),
            "dueAt": (datetime.combine(today_date,time(23,0)) - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M'),
            "endAt": (datetime.combine(today_date,time(23,0)) - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')
        }

        edit_res = editCard(
            token=token_str, board=app_wekan_board, new_swim=game_swimlane,
            new_list=doing_list, card=card_res['_id'], body=edit_data
        )

        checklist_data = {
            "title": f"{each['title']}-任务清单",
            "items": each['items']
        }

        checklist_res = createCheckList(
            token=token_str, board=app_wekan_board, 
            card=card_res['_id'], body=checklist_data
        )

    elif each['type'] == 'Thur' and today_date.weekday() == 3:

        card_res = createCard(
            token=token_str, board=app_wekan_board, list=doing_list, 
            author=app_wekan_author, title=f"{each['title']}-{today_str}",
            desc=f"{each['desc']}",swimlane=game_swimlane)
        
        edit_data = {
            "startAt": (datetime.combine(today_date,time(7,0)) - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M'),
            "dueAt": (datetime.combine(today_date,time(23,0)) - timedelta(hours=8) + timedelta(days=6)).strftime('%Y-%m-%d %H:%M'),
            "endAt": (datetime.combine(today_date,time(23,0)) - timedelta(hours=8) + timedelta(days=6)).strftime('%Y-%m-%d %H:%M')
        }

        edit_res = editCard(
            token=token_str, board=app_wekan_board, new_swim=game_swimlane,
            new_list=doing_list, card=card_res['_id'], body=edit_data
        )

        checklist_data = {
            "title": f"{each['title']}-任务清单",
            "items": each['items']
        }

        checklist_res = createCheckList(
            token=token_str, board=app_wekan_board, 
            card=card_res['_id'], body=checklist_data
        )

print("创建成功")