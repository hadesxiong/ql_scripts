# coding=utf8

# 父任务及子任务拼接方法
def taskParse(task_list:list) -> list:

    parent_ids = {
        item['id']: {
            'id_parent': item['id'],
            'content_parent': item['content'],
            'isDone':item['isDone'],
            'items':[]
        } for item in task_list if 'parentTaskId' not in item
    }


    for each in task_list:
        if 'parentTaskId' in each:

            parent_ids[each['parentTaskId']]['items'].append(
                {'id': each['id'],'content': each['content'],'isDone':each['isDone']}
            )
    
    return list(parent_ids.values())