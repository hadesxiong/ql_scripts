# coding=utf8
import os,requests

from .planParse import match_plan,parse_todolist,parse_content

# 读取环境变量
memos_url = os.getenv('app_memos_api_url')
memos_token = os.getenv('app_memos_token')

def plan_analysis(user: str, tag: str) -> dict:

    memos_headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {memos_token}'
    }

    memos_params = {
        'tag': tag,
        'creatorId': user
    }

    plan_rslt = requests.get(f'{memos_url}/api/v1/memo',
                             headers=memos_headers,params=memos_params)

    plan_list = plan_rslt.json()
    plan_data = match_plan(plan_list,tag[1])
    
    if plan_data.get('content'):
        
        plan_parse = parse_todolist(parse_content(plan_data['content'])[1:-1])
        
        result = {'summary':{}, 'undone_list':[]}
        count_all, done_all, undone_all = 0, 0, 0
        
        for k,v in plan_parse.items():
            task_count = len(v)
            task_done = len([task for task in v if task['complete']])
            task_undone = task_count - task_done

            count_all += task_count
            done_all += task_done
            undone_all += task_undone

            result['undone_list'] += [f"🧾{task['task']}" for task in v if not task['complete']]

        rate_all = f'{str(round(done_all/count_all,4)*100)}%'
        result['summary'] = {'count': count_all, 'done': done_all, 'undone': undone_all, 'rate': rate_all}

        return result

    else:
        return {}