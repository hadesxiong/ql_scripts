#!/usr/bin/env python3
# @desc HTML转纯文本清洗工具，去除标签、空行、多余换行；支持可选移除 script/style
# coding=utf8
import re
from bs4 import BeautifulSoup
import pandas as pd


def html_to_text(html_content, strip_special_tags=False, separator='\n'):
    if pd.isna(html_content):
        return ''
    try:
        soup = BeautifulSoup(str(html_content), 'html.parser')
        if strip_special_tags:
            for tag in soup(["script", "style"]):
                tag.decompose()
        text = soup.get_text(separator=separator, strip=True)
        if separator == ' ':
            text = re.sub(r'\s+', ' ', text)
        else:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
        return text
    except Exception as e:
        print(f'HTML解析错误: {e}')
        return str(html_content)


if __name__ == "__main__":
    test_html = "<div>测试<br/>内容</div>"
    print(html_to_text(test_html))