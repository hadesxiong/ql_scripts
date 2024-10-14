# coding=utf8
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from html2image import Html2Image

def get_page_height(html_path:str,browser_path:str) -> int:

    url = f'file://{html_path}'

    # 初始化webdriver
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")  # 禁用沙盒模式
    chrome_options.add_argument("--headless")  # 启用无头模式
    chrome_options.add_argument("--disable-dev-shm-usage")  # 禁用/dev/shm的使用
    service = Service(executable_path=browser_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # 打开目标网页
        driver.get(url)

        # 等待页面加载
        driver.implicitly_wait(5)  # 等待10秒，确保页面加载完成

        # 获取页面高度
        page_height = driver.execute_script(
            "return document.documentElement.scrollHeight"
        )
        print(f'the height of the page is: {page_height}px')

    except Exception as e:

        print(f'error occurred: {e}')

    finally:

        driver.quit()

    return page_height if page_height else None

def convert_html_png(html_file:str, css_file:str,ot_path:str,save_as:str,
                     img_width:int, img_height:int):
    
    hti = Html2Image(
        custom_flags=[
            '--virtual-time-budget=5',
            '--hide-scrollbars',
            '--no-sandbox'],
        output_path=ot_path)
    
    hti.screenshot(
        html_file=html_file, css_file=css_file,
        save_as=save_as, size=(int(img_width),int(img_height))
    )