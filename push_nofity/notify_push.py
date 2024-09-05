# codint=utf8
import os,requests,base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes

# 登陆密码加密
# 获取用户信息
app_nfy_user = os.getenv('app_nfy_user')
app_nfy_pwd = os.getenv('app_nfy_pwd')
app_nfy_key = os.getenv('app_nfy_key')
app_nfy_iv = os.getenv('app_nfy_iv')

# 获取应用地址
app_nfy_url = os.getenv('app_nfy_url')

# 转换为字节
key = app_nfy_key.encode('utf-8')
iv = app_nfy_iv.encode('utf-8')

# 创建加密器实例
cipher =AES.new(key, AES.MODE_CBC, iv)

# 对密码进行填充并加密
padded_password = pad(app_nfy_pwd.encode('utf-8'), AES.block_size)
encrypted_password = cipher.encrypt(padded_password)

# 将加密后的数据编码为base64字符串
encrypted_data = base64.b64encode(encrypted_password).decode('utf-8')

# 登陆获取token
login_url = f'{app_nfy_url}/user/userLogin'
login_data = {
    'username': app_nfy_user,
    'userpwd': encrypted_data
}

login_rslt = requests.request('POST',login_url,data=login_data)
print(login_rslt)
print(login_rslt.json()['token'])

# 消息推送
msg_url = f'{app_nfy_url}/message/sendMessage'
