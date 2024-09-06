# coding=utf8
import json,os,sys,datetime
# 添加引用路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from app_memos.planAnalysis import plan_analysis
from app_nfyserver.userLogin import user_login
from app_nfyserver.notifyPush import notify_push

# 读取目标用户
memos_user = json.loads(os.getenv('app_memos_user'))

# 获取推送频道信息
app_nfy_chnl_bark = f'chnl_{os.getenv("app_nfy_chnl_barkDaily")}'
app_nfy_chnl_ntfy = f'chnl_{os.getenv("app_nfy_chnl_ntfyDaily")}'

# 获取推送模板信息
app_nfy_tmpl_barkPC = f'tmpl_{os.getenv("app_nfy_tmpl_barkPC")}'
app_nfy_tmpl_ntfyPC = f'tmpl_{os.getenv("app_nfy_tmpl_ntfyPC")}'

# 读取推送功能用户名
app_nfy_user = os.getenv('app_nfy_user')
app_nfy_pwd = os.getenv('app_nfy_pwd')

app_nfy_token = user_login(app_nfy_user,app_nfy_pwd)

for each_user in memos_user:

    usr_week_plan = plan_analysis(str(each_user['creatorId']),'每周计划')
    print(usr_week_plan)
    usr_day_plan = plan_analysis(str(each_user['creatorId']),'每日计划')
    print(usr_day_plan)

    today = datetime.date.today()
    week_day = ['一','二','三','四','五','六','日'][today.weekday()]
    days_remain = 6 - today.weekday()

    pc_nfy_data = {
        'today_date': today.strftime("%Y-%m-%d"),
        'week_day': week_day,
        'day_remain': days_remain,
        'weektask_count': usr_week_plan['summary']['count'],
        'weektask_done': usr_week_plan['summary']['done'],
        'weekdone_rate': usr_week_plan['summary']['rate'],
        'weektasklist_undone': ('\n').join(usr_week_plan['undone_list']),
        'daytask_count': usr_day_plan['summary']['count'],
        'daytask_done': usr_day_plan['summary']['done'],
        'daydone_rate': usr_day_plan['summary']['rate'],
        'daytasklist_undone':  ('\n').join(usr_day_plan['undone_list']),
    }

    print(pc_nfy_data)

    if each_user.get('rcv').get('bark'):

        bark_push_res = notify_push(
            token = app_nfy_token, channel = app_nfy_chnl_bark,
            template = app_nfy_tmpl_barkPC, tmpl_type = 1,
            rcv_type = 'single', rcv_target = each_user.get('rcv').get('bark'),
            msg_dict = pc_nfy_data, call_from = 'ql_scripts_daily'
        )

        print(bark_push_res)

    if each_user.get('rcv').get('ntfy'):

        ntfy_push_res = notify_push(
            token = app_nfy_token, channel = app_nfy_chnl_ntfy,
            template = app_nfy_tmpl_ntfyPC, tmpl_type = 2,
            rcv_type = 'single', rcv_target = each_user.get('rcv').get('ntfy'),
            msg_dict = pc_nfy_data, call_from = 'ql_scripts_daily'
        )

        print(ntfy_push_res)