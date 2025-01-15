# coding=utf8
import json,os,sys,datetime,requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from app_tb.tbApi import getToken, queryTask
from app_tb.tbParse import taskParse
from app_nfyserver.userLogin import user_login
from app_nfyserver.notifyPush import notify_push

# 读取环境变量
app_tb_gameProj = os.getenv('app_tb_gameProj')

# 获取推送频道信息
app_nfy_chnl_bark = f'chnl_{os.getenv("app_nfy_chnl_barkDaily")}'
app_nfy_chnl_ntfy = f'chnl_{os.getenv("app_nfy_chnl_ntfyDaily")}'

# 获取推送模板信息
app_nfy_tmpl_barkPC = f'tmpl_{os.getenv("app_nfy_tmpl_barkPC")}'
app_nfy_tmpl_ntfyPC = f'tmpl_{os.getenv("app_nfy_tmpl_ntfyPC")}'

# 读取推送功能用户名
app_nfy_user = os.getenv('app_nfy_user')
app_nfy_pwd = os.getenv('app_nfy_pwd')

# 登陆获取token
app_tb_token = getToken()
app_nfy_token = user_login(app_nfy_user,app_nfy_pwd)

proj_list = [app_tb_gameProj]

# 查询今日未完成的任务数量
date_params = {'q': 'dueDate <= endOf(d)'}
date_result = { item: [] for item in proj_list}
for each_proj in proj_list:

    date_res = queryTask(app_tb_token, each_proj, date_params)
    date_dict = date_res['result']
    date_taskList = taskParse(date_dict)
    date_result[each_proj] = date_taskList

date_result_flatten = [item for sublist in date_result.values() for item in sublist]

# 查询本周未完成的任务数量
week_params = {'q': 'dueDate <= endOf(w)'}
week_result = { item: [] for item in proj_list}
for each_proj in proj_list:

    week_res = queryTask(app_tb_token, each_proj, week_params)
    week_dict = week_res['result']
    week_taskList = taskParse(week_dict)
    week_result[each_proj] = week_taskList

week_result_flatten = [item for sublist in week_result.values() for item in sublist]

today = datetime.date.today()
week_day = ['一','二','三','四','五','六','日'][today.weekday()]
days_remain = 6 - today.weekday()

datetask_count = sum(
    1 if len(item['items']) == 0 else len(item['items']) 
    for item in date_result_flatten
)

datetask_done = sum(
    item['isDone'] if len(item['items']) == 0 else sum(subitem['isDone'] for subitem in item['items'])
    for item in date_result_flatten
)

weektask_count = sum(
    1 if len(item['items']) == 0 else len(item['items']) 
    for item in week_result_flatten
)

weektask_done = sum(
    item['isDone'] if len(item['items']) == 0 else sum(subitem['isDone'] for subitem in item['items'])
    for item in week_result_flatten
)

pc_nfy_data = {
    'today_date': today.strftime("%Y-%m-%d"),
    'week_day': week_day,
    'day_remain': days_remain,
    'weektask_count': weektask_count,
    'weektask_done': weektask_done,
    'weekdone_rate': str(round(weektask_done/weektask_count*100,2))+'%',
    'weektasklist_undone': '请打开memos或者tb进行查看',
    'daytask_count': datetask_count,
    'daytask_done': datetask_done,
    'daydone_rate': str(round(datetask_done/datetask_count*100,2))+'%',
    'daytasklist_undone': '请打开memos或者tb进行查看'
}
print(pc_nfy_data)

# 读取目标用户
memos_user = json.loads(os.getenv('app_memos_user'))

for each_user in memos_user:

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

app_tb_projDict = json.loads(os.getenv('app_tb_projDict'))
# 创建memos检查清单
content = f'### 【{today.strftime("%Y-%m-%d")}】本周任务清单完成情况检视'
for key,value in week_result.items():

    content +='\n\n'
    content += f'#### {app_tb_projDict[key]} \n\n'

    for each in value:
        content += f'- [x] {each["content_parent"]}\n' if each['isDone'] else f'- [ ] {each["content_parent"]}\n'

        if len(each['items']) > 0:
            for each_sub in each['items']:
                content += f'  - [x] {each_sub["content"]}\n' if each_sub['isDone'] else f'  - [ ] {each_sub["content"]}\n'

content +=f'\n#每周计划'

# 读取环境变量
memos_url = os.getenv('app_memos_api_url')
memos_token = os.getenv('app_memos_token')

memos_headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {memos_token}'
}

memos_data = {'content': content, 'visibility':'PUBLIC'}

memos_rslt = requests.post(f'{memos_url}/api/v1/memo',json = memos_data, headers= memos_headers)

if memos_rslt.status_code == 200:
    print('创建成功')
else:
    print('创建失败')