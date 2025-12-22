# coding=utf8
import os,json,hashlib,requests

from base64 import b64decode

try:
    from Cryptodome.Cipher import AES
    from Cryptodome.Util.Padding import unpad
    from Cryptodome.Protocol.KDF import PBKDF2
except ImportError:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import unpad
    from Crypto.Protocol.KDF import PBKDF2

cc_url = os.getenv('app_cookiecloud_url')
cc_uuid = os.getenv('app_cookiecloud_uuid')
cc_key = os.getenv('app_cookiecloud_key')

website_env_map = json.loads(os.getenv('app_cookiecloud_websitemap'))

cookies = requests.get(f'{cc_url}/get/{cc_uuid}').json()

key = hashlib.md5(f'{cc_uuid}-{cc_key}'.encode()).hexdigest()[:16].encode()

ciphertext = cookies['encrypted']
encrypted_data = b64decode(ciphertext)
salt = encrypted_data[8:16]
ct = encrypted_data[16:]

key_iv = b''
prev = b''

while len(key_iv) < 48:
    prev = hashlib.md5(prev+key+salt).digest()
    key_iv += prev

_key = key_iv[:32]
_iv = key_iv[32:48]

cipher = AES.new(_key, AES.MODE_CBC, _iv)
pt = unpad(cipher.decrypt(ct), AES.block_size)

decrypted_data = json.loads(pt.decode('utf-8'))

cookie_data_all = json.dumps(decrypted_data['cookie_data'],ensure_ascii=False,indent=2)
cookie_data = json.loads(cookie_data_all)

env_contents = {}

for w,c in cookie_data.items():
    formatted_cookies = []
    for cookie in c:
        formatted_cookies.append(f'{cookie["name"]}={cookie["value"]}')
        # print(cookie["name"],cookie["value"])

    formatted_cookies = list(set(formatted_cookies))
    # print(formatted_cookies)
    cookie_result = '; '.join(formatted_cookies)    
    print(cookie_result)
    print(w)

    website_lower = w.lower()
    if website_lower in website_env_map:
        env_var = website_env_map[website_lower]
        # env_contents.append(f'{env_var}={cookie_result}')
        env_contents[env_var] = cookie_result
        print(f'已处理{w}的cookie')

print(env_contents)

output_path = '/ql/data/scripts/output/rsshub/env/.env'

if env_contents:
    with open(output_path, 'w', encoding='utf-8') as f:
        for each_key,each_value in env_contents.items():
            f.write(f'{each_key}={each_value}'+'\n')

print('已将所有cookie写入')