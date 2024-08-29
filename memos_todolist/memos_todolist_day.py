# coding=utf8
import requests,os,json
from requests.auth import AuthBase
from datetime import datetime, timedelta

# 定义验证方法
class BearAuth(AuthBase):

    def __init__(self,token):
        self.token = token

    def __call__(self,r):
        r.headers['Authorization'] = f'Bearer {self.token}'
        return r
    
# 定义markdown写法
def CreateMd(plan_dict:dict) -> str:

    md_str = ""

    # 日期配置
    date_today = datetime.now()
    remain_days = (datetime(date_today.year,12,31) - date_today).days

    week_of_year = date_today.isocalendar()[1]
    day_of_week = date_today.weekday()+1

    # md_str += f'''
    #     今天是{date_today.year}年{date_today.month}月{date_today.day}日,
    #     第{week_of_year}周的第{day_of_week}天,
    #     距离今年结束还有{remain_days},
    #     星光不负赶路人，愿你每天都离目标更近一点/n
    # '''

    md_str += f'#### {date_today.year}年{date_today.month}月{date_today.day}日-每日待办事项清单\n\n'

    try:
        daily_plan_data = json.loads(plan_dict)
        for k,v in daily_plan_data.items():
            md_str += f'#### {k}\n\n'
            for vs in v:
                md_str += f'- [ ] {vs}\n'
            md_str += '\n'

    except Exception as e:
        print(f'发生错误:\n{e}')
        raise

    md_str += '#每日计划'
    
    return md_str


# 读取环境变量
memos_url = os.getenv('memos_url')
memos_token = os.getenv('memos_token')
memos_daily_plan = os.getenv('memos_daily_plan')

plan_content = CreateMd(memos_daily_plan)
# print(plan_content)

# 创建链接
sessions = requests.Session()
sessions.auth = BearAuth(memos_token)

plan_data = {'content': plan_content,'visibility':'PUBLIC'}
memos_rslt = sessions.post(f'{memos_url}/api/v1/memo',json=plan_data)

if memos_rslt.status_code == 200:
    print('创建成功')
else:
    print('创建失败')