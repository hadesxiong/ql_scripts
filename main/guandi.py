# coding=utf8
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

def fetch_html(number):
    
    url = f'https://www.nongli.com/item5/guandi/{number}.html'
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")  # 禁用沙盒模式
    chrome_options.add_argument("--headless")  # 启用无头模式
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-dev-shm-usage")  # 禁用/dev/shm的使用

    # 初始化webdriver
    chrome_driver_path = '/usr/bin/chromedriver'
    chrome_service = Service(executable_path=chrome_driver_path)

    # 初始化webdriver
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    
    try:
        driver.get(url)
        # 等待页面加载完成，这里等待10秒，你可以根据实际情况调整等待时间
        driver.implicitly_wait(5)
        html = driver.page_source
        # print(html)
        return html
    finally:
        driver.quit()
    
def extract_content(html):

    soup = BeautifulSoup(html, 'html.parser')
    biaoti_div = soup.find('div', class_='kingkuang')
    if biaoti_div:
        # return biaoti_div.get_text(strip=True)
        biaoti_parts = biaoti_div.get_text(strip=True).split( )
        if len(biaoti_parts) > 1:
            biaoti_content = ','.join(biaoti_parts[1:])
        else:
            biaoti_content = biaoti_div.get_text(strip=True)
    else:
        # return '未找到class为kingkuang的div标签'
        biaoti_content = '未找到class为kingkuang的div标签'
    
    tujie_div = soup.find('div', class_='tujie_main')
    if tujie_div:
        p_tag = tujie_div.find('p')
        if p_tag:
            # return p_tag.get_text(strip=True)
            tujie_content = p_tag.get_text(strip=True)
        else:
            # return "未找到p标签"
            tujie_content = "未找到p标签"
    else:
        # return "未找到class为'tujie_main'的div标签"
        tujie_content = "未找到class为'tujie_main'的div标签"

    return biaoti_content, tujie_content
    
for number in range(1,101):
    html = fetch_html(number)
    if html:
        biaoti_content, tujie_content = extract_content(html)
        with open('guandi_sign.txt', 'a', encoding='utf-8') as file:
            file.write(f'{biaoti_content}:{tujie_content}\n')
        print(f"第{number}签的标题内容：{biaoti_content}_{tujie_content}")