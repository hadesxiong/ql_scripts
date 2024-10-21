# coding=utf8
import json,requests

# 获取access_toekn
def get_wx_token(appid:str,secret:str) -> str:

    wx_token_url = 'https://api.weixin.qq.com/cgi-bin/token'

    wx_params = {
        "grant_type": "client_credential",
        "appid": appid,
        "secret": secret
    } 


    wx_response = requests.get(wx_token_url,params=wx_params)

    res_data = json.loads(wx_response.text)
    print(res_data)
    if res_data.get('access_token'):
        return {'result':True,'token':res_data['access_token']}
    else:
        return {'result':False,'err':res_data['errmsg']}
    
# 上传永久素材
def upload_wx_material(token:str='',file_type:str='image',
                       file_name:str='',file_path:str='',) -> dict:
    
    wx_upload_url = f'https://api.weixin.qq.com/cgi-bin/material/add_material?access_token={token}&type={file_type}'

    with open(file_path,'rb') as file_obj:

        file_data = {'media':(file_name,file_obj)}

        upload_res = requests.post(wx_upload_url,files=file_data)

    res_data = json.loads(upload_res.text)

    if res_data.get('media_id'):
        return {'media_id':res_data['media_id'],'url':res_data['url']}
    else:
        return {'err_msg':res_data['errmsg']}

# 获取素材列表
def get_wx_material(token:str='', type:str='image',
                    offset:int=0, count:int=20) -> dict:
    
    wx_material_url = f'https://api.weixin.qq.com/cgi-bin/material/batchget_material?access_token={token}'

    query_args = {
        'type': type,
        'offset': offset,
        'count': count
    }
    material_res = requests.post(wx_material_url,data=query_args)

    return json.loads(material_res.text)

# 新建图文草稿
def create_md_draft(token:str='',article_list:list=[]) -> dict:

    wx_draft_url = f'https://api.weixin.qq.com/cgi-bin/draft/add?access_token={token}'

    # 标准化处理article对象
    draft_list, fail_list = [], []

    for each in article_list:
        
        # 格式化
        each_format = {
            'title': each.get('title',None),
            'author': each.get('author','Nero.Feng.'),
            'digest': each.get('digest',None),
            'content': each.get('content',None),
            'content_source_url': each.get('content_source_url',None),
            'thumb_media_id': each.get('thumb_media_id',None),
            'need_open_comment': each.get('need_open_comment',0),
            'only_fans_can_comment': each.get('only_fans_can_comment',0),
            'pic_crop_235_1': each.get('pic_crop_235_1','0_0_1_1'),
            'pci_crop_1_1':each.get('pic_crop_1_1','0_0_1_1')
        }

        # 检验
        if each_format.get('title') and each_format.get('content') and each_format.get('thumb_media_id'):
            draft_list.append(each_format)

        else:
            fail_list.append(each_format)

    if len(draft_list) < 1:

        return {'result':False, 'errmsg':'no data verified pass.'}
    
    else:

        draft_res = requests.post(wx_draft_url,data=json.dumps({'articles':draft_list}, ensure_ascii=False))
        # res_data = json.loads(draft_res.text)
        res_data = json.loads(draft_res.text.encode('latin1').decode('utf-8'))

        if res_data.get('media_id',None):
            return {'result':True, 'count_pass':len(draft_list), 'count_fail':len(fail_list),'media_id':res_data['media_id']}
        else:
            return {'result':False, 'count_pass':len(draft_list), 'count_fail':len(fail_list),'err_msg':res_data['errmsg']}
