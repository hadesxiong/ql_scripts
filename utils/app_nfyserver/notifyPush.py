# coding=utf8
import json,os,requests

def notify_push(
        token:str, channel:str, template:str, tmpl_type:int, 
        rcv_type: str, rcv_target: str, msg_dict:dict,
        call_from:str, args: dict = None) -> str:
    
    nfyserver_host = os.getenv('app_nfy_url')
    notify_url = f'http://{nfyserver_host}/message/sendMessage'

    notify_data = {
        "channel": channel,
        "template": template,
        "message": {
            "type": tmpl_type,
            "receive": rcv_type,
            "target": rcv_target,
            "data": msg_dict
        },
        "call": call_from
    }

    if args:
        notify_data['message']['args'] = args

    notify_headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    notify_res = requests.post(url=notify_url,data=json.dumps(notify_data),headers=notify_headers)

    return {
        'code': notify_res.status_code,
        'text': notify_res.text
    }