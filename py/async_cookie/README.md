# CookieCloud-Cookie同步脚本 async_cookie

## 基础信息
- 脚本入口：`py/async_cookie/main.py`
- 开发语言：Python3
- 依赖公共组件：
  1. `components/py/aes_crypt.py` CookieCloud AES-CBC 解密
- 青龙推荐定时：0 6 * * *

## 功能简介
1. 从 CookieCloud 服务器拉取加密 Cookie 数据；
2. AES-CBC 解密，还原明文 Cookie；
3. 按域名映射写入 `.env` 文件。

## 运行产出
1. 文件产出：`output/async_cookie/.env`

## 环境变量配置
复制 config.tpl.env 内变量填入青龙面板环境变量；
域名映射 JSON 格式示例：`{"rsshub.app": "COOKIE_RSSHUB"}`

## 本地调试运行
```bash
cd py/async_cookie
python3 main.py
```

## 更新记录
2026-06-24：脚本初始化