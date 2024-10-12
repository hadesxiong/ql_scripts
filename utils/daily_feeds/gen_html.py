# coding=utf8
# from html2image import Html2Image
# from datetime import datetime

# hti = Html2Image(custom_flags=['--virtual-time-budget=360','--hide-scrollbars','--no-sandbox'])

# def generate_html(news_dict)

from string import Template
from datetime import datetime

with open('./template/tmpl_tech.html','r',encoding='utf-8') as html_file:
    html_template = Template(html_file.read())

container_template = '''
    <div class="container">
        <div class="title_con">
            <div class="line_con"><div class="title_line"></div></div>
            <div class="title_main">${title_main}</div>
        </div>
        ${ban_data}
        <div class="feed_con">
        ${ban_before_data}
            <div class="feed_abst">
                ${feed_abst}
            </div>
        ${ban_after_data}
        </div>
        <hr class="split_line">
    </div>
'''

news_data = [
    {
        "title_main": "vivo X200 系列手机定档 10 月 14 日发布",
        "background_image": "https://cdn.pingwest.com/portal/2024/09/30/WdRd7ABNt45A2f3zm7NQf7t4McCMFfB5.jpg?x-oss-process=style/article-body",
        "feed_abst":"vivo vivo X200 系列手机定档 10 月 14 日发布 vivo 今天官宣 X200 系列手机定档10 月 14 日 19:00 亮相，发布会将在北京水立方举行。"
    },
    {
        "title_main": "阿里妈妈开源全新 AI 图像修复模型 FLUX-Controlnet-Inpainting",
        "background_image": "https://cdn.pingwest.com/portal/2024/09/30/ttQWPZB7DFsQ66cH5iaa332sH2wrSBW7.jpg?x-oss-process=style/article-body",
        "feed_abst": "阿里妈妈 阿里妈妈开源全新 AI 图像修复模型 FLUX-Controlnet-Inpainting 阿里妈妈创意团队宣布开源FLUX-Controlnet-Inpainting AI 图像修复模型。"
    },
    {
        "title_main": "联想消费国庆新品嘉年华：联想YOGA AI元启新品引领移动办公与创作新潮流",
        "background_image":"https://cdn.pingwest.com/portal/2024/09/30/portal/2024/09/30/WSBdC4JH63276i7C7n15G53kYHz64Sky?x-oss-process=style/article-body",
        "feed_abst": "联想 联想消费国庆新品嘉年华：联想YOGA AI元启新品引领移动办公与创作新潮流"
    },
    {
        "title_main": "Meta的新眼镜Orion，就是下一代消费级AI设备的“GPT3时刻”系",
        "background_image": "https://cdn.pingwest.com/portal/2024/09/30/portal/2024/09/30/x8NKpzGady63a660WDEN48Jicnz776Ey.png?x-oss-process=style/article-body",
        "feed_abst": "Meta的新眼镜Orion，就是下一代消费级AI设备的“GPT3时刻” 路线已经清晰，一切都会更快发生"
    }
]

container_data = ''

for index,item in enumerate(news_data):

    ban_data = ''
    ban_before_data = ''
    ban_after_data = ''

    if index == 0:

        ban_data = '''
        <div class=\"ban first_ban\" style=\"background-image:url({image});\"></div>\n
        '''.format(image=item['background_image'])

    elif index != 0 and index % 2 == 0:
        ban_after_data = '''
        <div class=\"ban\" style=\"background-image:url({image});\"></div>\n
        '''.format(image=item['background_image'])
    else:
        ban_before_data = '''
        <div class=\"ban\" style=\"background-image:url({image});\"></div>\n
        '''.format(image=item['background_image'])

    each_container = Template(container_template).substitute(
        title_main = item['title_main'],
        feed_abst = item['feed_abst'],
        ban_data = ban_data,
        ban_before_data = ban_before_data,
        ban_after_data = ban_after_data
    )

    container_data += each_container

today_dt = datetime.now().strftime('%Y.%m.%d')

html_output = html_template.substitute(
    daily_info=f'[{today_dt}]FeedsDaily#1',
    container_data=container_data
)

with open('output.html','w',encoding='utf-8') as file:

    file.write(html_output)

print('done')