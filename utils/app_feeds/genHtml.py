# coding=utf8
import json,os,requests

from string import Template
from datetime import datetime
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse,unquote

def get_img_extension(url):

    # 解析URL并提取路径
    parsed_url = urlparse(url)
    path = unquote(parsed_url.path)  # 解码路径，以防有特殊字符
    
    # 提取文件扩展名
    _, ext = os.path.splitext(path)
    return ext.lower()

def gen_html(html_tmpl:Template,
             con_tmpl:str,
             feeds_data:list,
             target_dir:str):

    container_data = ''

    sorted_data = sorted(feeds_data, key=lambda x: (x['background_image'] is not None, x['background_image']), reverse=True)[:10]

    today_dt = datetime.now().strftime('%Y_%m_%d')

    for index,item in enumerate(sorted_data):

        ban_data = ''
        ban_before_data = ''
        ban_after_data = ''
    
        if item.get('background_image',None):

            try:

                img_response = requests.get(item.get('background_image'))

                if img_response.status_code == 200:
                    img_data = BytesIO(img_response.content)
                    img_file = Image.open(img_data)
                    print(img_file)
                    if img_file.mode == 'RGB':
                        img_file = img_file.convert('RGBA')
                    print(img_file)
                    # 调整尺寸
                    origin_width, origin_height = img_file.size
                    new_width = 824
                    new_height = int((origin_height/origin_width)*new_width)
                    resized_img = img_file.resize((new_width, new_height))
                    
                    if get_img_extension(item.get('background_image')) == '':
                        file_ext = 'png'
                    else:
                        file_ext = get_img_extension(item.get('background_image'))[1:]
                    # img_filePath = f'{target_dir}/img/{item["entry_id"]}.{file_ext.lower()}'
                    img_filePath = f'{target_dir}/img/{today_dt}_{index}.{file_ext.lower()}'

                    if file_ext in ['jpg', 'jpeg','png','JPG','JPEG','PNG']:
                        resized_img.save(img_filePath,'PNG',quality=85)
                        
                    else:
                        resized_img.save(img_filePath,file_ext.upper())

                    item['background_image'] = img_filePath

                else:
                    item['background_image'] = None

            except Exception as e:
                item['background_image'] = None
                print('save fail',item['background_image'],item['entry_id'])

        
        if index == 0 and item.get('background_image',None):

            ban_data = '''
            <div class=\"ban first_ban\" style=\"background-image:url(file://{image});\"></div>\n
            '''.format(image=item['background_image'])

        elif index != 0 and index % 2 == 0 and item.get('background_image',None):

            ban_after_data = '''
            <div class=\"ban\" style=\"background-image:url(file://{image});\"></div>\n
            '''.format(image=item['background_image'])

        elif item.get('background_image',None):

                ban_before_data = '''
                <div class=\"ban\" style=\"background-image:url(file://{image});\"></div>\n
                '''.format(image=item['background_image'])

        each_container = Template(con_tmpl).substitute(
            title_main = item['title_main'],
            feed_abst = item['feed_abst'],
            ban_data = ban_data,
            ban_before_data = ban_before_data,
            ban_after_data = ban_after_data
        )

        container_data += each_container

    today_dt = datetime.now().strftime('%Y_%m_%d')

    target_path = f'{target_dir}/{today_dt}_tech_feeds.html'

    html_output = html_tmpl.substitute(
        container_data = container_data
    )

    with open(target_path,'w',encoding='utf-8') as file:
        file.write(html_output)

    print(f'{today_dt}_tech_feeds.html has been created.')

    return target_path

def format_entry_list(entry_list:list):

    format_list = []

    # 读取来源映射字典
    feed_dict = json.loads(os.getenv('app_fresh_target_dict'))

    for each in entry_list:

        entry_content = each['content'].decode('utf-8')
        entry_dict = {
            'entry_id': f"entry_{each['id']}",
            'title_main': each['title'],
            'feed_abst': '',
            'background_image': ''
        }

        soup = BeautifulSoup(entry_content,'html.parser')
        entry_dict['feed_abst'] = soup.get_text(strip=True)

        try:
            if soup.find():
                images = soup.find_all('img')
                src_list = [img.get('src') for img in images if img.get('src')]
                entry_dict['background_image'] = src_list[0]

            else:
                entry_dict['background_image'] = None

        except Exception as e:
            print(e)
            entry_dict['background_image'] = None

        format_list.append(entry_dict)

    return format_list