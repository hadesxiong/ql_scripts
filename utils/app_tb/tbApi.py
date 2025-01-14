# coding=utf8
import os,json,requests

# 读取环境变量
tb_url = os.getenv('app_tb_url')
tb_tenant = os.getenv('app_tb_tenant')
tb_creator = os.getenv('app_tb_creator')
tb_appId = os.getenv('app_tb_appid')
tb_appSecret = os.getenv('app_tb_secret')

# 获取应用授权
def getToken() -> str:

    token_url = f'{tb_url}/appToken'
    token_data = {'appId': tb_appId, 'appSecret': tb_appSecret}
    token_res = requests.post(token_url, data=token_data)

    return token_res.json().get('appToken', 'blank')

# 创建任务
def createTask(token:str, data: dict) -> dict:

    task_url = f'{tb_url}/v3/task/create'
    task_headers = {
        'Authorization': f'Bearer {token}',
        'X-Tenant-Id': tb_tenant,
        'X-Tenant-Type': 'organization',
        'x-operator-id': tb_creator
    }
    task_res = requests.post(task_url, headers=task_headers,data=data)
    return task_res.json()

# 查询任务
def queryTask(token:str, proj:str) -> dict:

    task_url = f'{tb_url}/v3/project/{proj}/task/query'
    task_headers = {
        'Authorization': f'Bearer {token}',
        'X-Tenant-Id': tb_tenant,
        'X-Tenant-Type': 'organization',
        'x-operator-id': tb_creator
    }

    task_res = requests.get(task_url, headers=task_headers)
    return task_res.json()
