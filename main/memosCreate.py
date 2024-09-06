# coding=utf8
import os,sys,requests
# 添加引用路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from app_memos.planParse import create_plan_daily

# 读取环境变量
memos_url = os.getenv('app_memos_api_url')
memos_token = os.getenv('app_memos_token')
memos_daily_plan = os.getenv('app_memos_dailyPlan')

plan_content = create_plan_daily(memos_daily_plan)

memos_headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {memos_token}'
}

memos_data = {'content': plan_content, 'visibility':'PUBLIC'}

memos_rslt = requests.post(f'{memos_url}/api/v1/memo',json = memos_data, headers= memos_headers)

if memos_rslt.status_code == 200:
    print('创建成功')
else:
    print('创建失败')