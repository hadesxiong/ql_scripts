#!/usr/bin/env python3
# @desc CookieCloud AES-CBC 解密组件，实现 PBKDF2 密钥派生 + AES 解密
# coding=utf8
import json
import hashlib
from base64 import b64decode

try:
    from Cryptodome.Cipher import AES
    from Cryptodome.Util.Padding import unpad
    from Cryptodome.Protocol.KDF import PBKDF2
except ImportError:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import unpad
    from Crypto.Protocol.KDF import PBKDF2


def cookiecloud_decrypt(encrypted_data: str, uuid: str, key: str) -> dict:
    """
    解密 CookieCloud 加密数据
    :param encrypted_data: base64 编码的加密字符串
    :param uuid: CookieCloud UUID
    :param key: CookieCloud 密钥
    :return: 解密后的 JSON 字典
    """
    key_material = hashlib.md5(f'{uuid}-{key}'.encode()).hexdigest()[:16].encode()
    encrypted_bytes = b64decode(encrypted_data)
    salt = encrypted_bytes[8:16]
    ct = encrypted_bytes[16:]

    # PBKDF2 派生 key + iv
    key_iv = b''
    prev = b''
    while len(key_iv) < 48:
        prev = hashlib.md5(prev + key_material + salt).digest()
        key_iv += prev

    _key = key_iv[:32]
    _iv = key_iv[32:48]

    cipher = AES.new(_key, AES.MODE_CBC, _iv)
    pt = unpad(cipher.decrypt(ct), AES.block_size)
    return json.loads(pt.decode('utf-8'))


if __name__ == "__main__":
    # 自测入口
    print("aes_crypt 组件加载成功")