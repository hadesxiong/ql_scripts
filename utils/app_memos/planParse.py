# coding=utf8
import json,re
from datetime import datetime, timedelta

# 转换md文档中的内容
def parse_content(content_str:str) -> list:

    return content_str.replace('\n\n','\n').split('\n')

# 转换标题的日期
def parse_date(date_str:str) -> list:

    date_list = date_str.split('-每')
    
    if '日' in date_list[1]:
        start_date = datetime.strptime(date_list[0].replace('#### ',''), '%Y年%m月%d日')
        return start_date

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

# 转换待处理项目
def parse_taskitem(task_item: str):

    # 检查任务是否完成
    if task_item.startswith("- [x] "):
        return task_item[5:].strip(), True
    elif task_item.startswith("- [ ] "):
        return task_item[5:].strip(), False
    else:
        return task_item.strip(), False  # 默认未完成
    
# 转换为todolist
def parse_todolist(items: list) -> dict:

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
            task, status = parse_taskitem(item)
            # 如果当前分类存在，添加任务到当前分类的列表中
            if current_category:
                result[current_category].append({
                    "task": task,
                    "complete": status
                })
    
    return result

# 找出符合符合当前日期的月/周/日计划
def match_plan(plan_list:list, type:str) -> dict:
    
    # 获取当前日期
    today_date = datetime.today()
    today_date  = today_date.replace(hour=0,minute=0,second=0,microsecond=0)
    # 处理plan_list，转化成日期区间
    date_list = list(
        map(lambda x: parse_date(parse_content(x['content'])[0]),plan_list)
    )

    if type == '日':
        for index,(start_date) in enumerate(date_list):
            if start_date.date() == today_date.date():
                return plan_list[index]
            else:
                pass
    
    elif type == '周' or type == '月':
        for index,(start_date,end_date) in enumerate(date_list):
            if start_date <= today_date <= end_date:
                return plan_list[index]
            else:
                return {}
            
# 创建每日计划
def create_plan_daily(plan_dict:dict) -> str:

    md_str = ''

    # 配置日期
    date_today = datetime.now()
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
