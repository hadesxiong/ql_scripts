# coding=utf8
import requests,os,json,re
from requests.auth import AuthBase
from datetime import datetime,timedelta

# 定义验证方法
class BearAuth(AuthBase):

    def __init__(self,token):
        self.token = token

    def __call__(self,r):
        r.headers['Authorization'] = f'Bearer {self.token}'
        return r
    
# 定义解析方法
# 1. 替换双层换行的为单层换行，并拆分成为数组
# 2. 提取标题部分进行日期解析
# 3. 将剩余内容分类进行结构化

def parseMdContent(content_str:str) -> list:
    
    return content_str.replace('\n\n','\n').split('\n')

def parseDate(date_str:str) -> list:

    date_list = date_str.split('-每')
    
    if '日' in date_list[1]:
        start_date = datetime.strptime(date_list[0].replace('#### ',''), '%Y年%m月%d日')
        return [start_date]

    elif '周' in date_list[1]:
        week_pattern = re.compile(r'(\d{4}年\d{1,2}月\d{1,2}日)-(\d{4}年\d{1,2}月\d{1,2}日)')
        match = week_pattern.match(date_list[0].replace('#### ',''))
        start_date_str, end_date_str = match.groups()
        start_date = datetime.strptime(start_date_str, '%Y年%m月%d日')
        end_date = datetime.strptime(end_date_str, '%Y年%m月%d日')
        return [start_date, end_date]
    
    elif '月' in date_list[1]:
        month_pattern = re.compile(r'(\d{4})年(\d{1,2})月')
        match = month_pattern.match(date_list[0].replace('#### ',''))
        year, month = int(match.group(1)), int(match.group(2))
        start_date = datetime(year, month, 1)
        last_day = start_date + timedelta(days=(31 - start_date.day))
        while last_day.month != month:
            last_day = last_day - timedelta(days=1)
        end_date = datetime(year, month, last_day.day)
        return [start_date, end_date]

def parseTaskItem(task_item:str):
    
    # 检查任务是否完成
    if task_item.startswith("- [x] "):
        return task_item[5:].strip(), True
    elif task_item.startswith("- [ ] "):
        return task_item[5:].strip(), False
    else:
        return task_item.strip(), False  # 默认未完成
    
def parseTodoList(items:list) -> dict:
    
    # 初始化结果字典
    result = {}
    # 初始化当前分类
    current_category = None
    # 遍历数组中的每个元素
    for item in items:
        # 去除前后空白字符
        item = item.strip()
        
        # 检查是否是分类标题
        if item.startswith("#### ") or item.startswith("## ") or item.startswith("# "):
            # 提取分类名称，去掉前后的井号和空格
            current_category = item[5:].strip()
            # 确保当前分类在结果字典中有对应的列表
            if current_category not in result:
                result[current_category] = []
        elif item.startswith("- ["):
            # 检查是否是任务项
            task, status = parseTaskItem(item)
            # 如果当前分类存在，添加任务到当前分类的列表中
            if current_category:
                result[current_category].append({
                    "task": task,
                    "complete": status
                })
    
    return result

# 找出符合符合当前日期的月/周/日计划
def matchPlanData(plan_list:list, type:str) -> dict:
    
    # 获取当前日期
    today_date = datetime.today()
    # 处理plan_list，转化成日期区间
    date_list = list(
        map(lambda x: parseDate(parseMdContent(x['content'])[0]),plan_list)
    )

    if type == '日':
        for index,(start_date) in enumerate(date_list):
            if start_date == today_date:
                return plan_list[index]
            else:
                return plan_list[0]
    
    elif type == '周' or type == '月':
        for index,(start_date,end_date) in enumerate(date_list):
            if start_date <= today_date <= end_date:
                return plan_list[index]
            else:
                return plan_list[0]

# 分析未完成事项
def analysisPlan(plan_dict:dict) -> dict:
    
    # 定义一个结果用来返回
    result = {'summary':{},'detail':{},'undone':{}}

    count_all, done_all, undone_all = 0, 0, 0

    for k,v in plan_dict.items():
        task_count = len(v)
        task_done = len([task for task in v if task['complete']])
        task_undone = task_count - task_done
        
        count_all += task_count
        done_all += task_done
        undone_all += task_undone

        result['detail'][k] = {'count':task_count,'done':task_done,'undone':task_undone, 'rate': f'{round(task_done/task_count,4)*100}%'}

        result['undone'][k] = [task['task'] for task in v if not task['complete']]

    result['summary']= {'count': count_all, 'done': done_all, 'undone': undone_all, 'rate': f'{round(done_all/count_all,4)*100}%'}

    return result

# 格式化分析结果
def formatAnalysis(analysis_dict:dict, type:str) -> str:

    return f'''
       您本{type}的完成进度如下：
       需要完成事项共计{analysis_dict['summary']['count']}项,
       已经完成事项共计{analysis_dict['summary']['done']}项,
       未完成事项共计{analysis_dict['summary']['undone']}项,
       完成率{analysis_dict['summary']['rate']}
    '''

# 读取环境变量
memos_url = os.getenv('memos_url')
memos_token = os.getenv('memos_token')

# 创建链接
sessions = requests.Session()
sessions.auth = BearAuth(memos_token)

# 获取每周计划完成情况
weekTag_params = {'tag':'每周计划'}
weekPlan_rslt = sessions.get(f'{memos_url}/api/v1/memo',params=weekTag_params)

weekPlan_list = weekPlan_rslt.json()

week_plan = matchPlanData(weekPlan_list,'周')
week_parse = parseTodoList(parseMdContent(week_plan['content'])[1:-1])
week_analysis = formatAnalysis(analysisPlan(week_parse),'周')

# 获取每日计划完成情况
dayTag_params = {'tag':'每日计划'}
dayPlan_rslt = sessions.get(f'{memos_url}/api/v1/memo',params=dayTag_params)

dayPlan_list = dayPlan_rslt.json()

day_plan = matchPlanData(dayPlan_list,'日')
day_parse = parseTodoList(parseMdContent(day_plan['content'])[1:-1])
day_analysis = formatAnalysis(analysisPlan(day_parse),'日')

print(week_analysis)
print(day_analysis)