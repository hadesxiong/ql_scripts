# coding=utf8
import json,os

from string import Template
from datetime import datetime
from bs4 import BeautifulSoup

def gen_html(html_tmpl:Template,
             con_tmpl:str,
             feeds_data:list,
             target_dir:str):

    container_data = ''

    for index,item in enumerate(feeds_data):

        ban_data = ''
        ban_before_data = ''
        ban_after_data = ''

        if index == 0:

            if item.get('background_image',None):

                ban_data = '''
                <div class=\"ban first_ban\" style=\"background-image:url({image});\"></div>\n
                '''.format(image=item['background_image'])

        elif index != 0 and index % 2 == 0:

            if item.get('background_image',None):

                ban_after_data = '''
                <div class=\"ban\" style=\"background-image:url({image});\"></div>\n
                '''.format(image=item['background_image'])
        else:

            if item.get('background_image',None):

                ban_before_data = '''
                <div class=\"ban\" style=\"background-image:url({image});\"></div>\n
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