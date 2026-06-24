#!/usr/bin/env python3
# @name CookieCloud Cookie同步脚本
# @desc 从 CookieCloud 拉取加密 Cookie，解密后写入 .env 文件
# @output output/async_cookie/.env
# coding=utf8
import json
import sys
import os
import requests
from pathlib import Path

BASE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_ROOT))

from components.py.aes_crypt import cookiecloud_decrypt
from components.py.env_loader import load_script_env

load_script_env(Path(__file__).parent)

def main():
    cc_url = os.getenv('app_cookiecloud_url', '').rstrip('/')
    cc_uuid = os.getenv('app_cookiecloud_uuid', '')
    cc_key = os.getenv('app_cookiecloud_key', '')
    website_env_map = json.loads(os.getenv('app_cookiecloud_websitemap', '{}'))

    if not cc_url or not cc_uuid or not cc_key:
        print("❌ 缺少必要环境变量：app_cookiecloud_url / uuid / key")
        sys.exit(1)

    # 1. 拉取加密数据
    resp = requests.get(f'{cc_url}/get/{cc_uuid}')
    resp.raise_for_status()
    encrypted = resp.json()['encrypted']

    # 2. 解密
    decrypted = cookiecloud_decrypt(encrypted, cc_uuid, cc_key)
    cookie_data = decrypted.get('cookie_data', {})

    # 3. 域名 → 环境变量映射
    env_contents = {}
    for domain, cookies in cookie_data.items():
        formatted = list(set(f'{c["name"]}={c["value"]}' for c in cookies))
        cookie_str = '; '.join(formatted)
        domain_lower = domain.lower()
        if domain_lower in website_env_map:
            env_contents[website_env_map[domain_lower]] = cookie_str
            print(f'✅ 已处理 {domain} → {website_env_map[domain_lower]}')

    # 4. 写入 .env 文件
    if not env_contents:
        print("⚠️ 没有匹配到任何 Cookie 映射，跳过写入")
        return

    output_path = BASE_ROOT / "output" / "async_cookie" / ".env"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for k, v in env_contents.items():
            f.write(f'{k}={v}\n')

    print(f'✅ 已将 {len(env_contents)} 个 Cookie 写入 {output_path}')

if __name__ == "__main__":
    main()