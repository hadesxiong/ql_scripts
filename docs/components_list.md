# 公共组件清单（自动生成，禁止手动修改）
> 本文件由 tools/gen_components_doc.py 生成，读取组件文件第二行 `@desc` 作为介绍

## Python组件
- [aes_crypt.py](./components\py\aes_crypt.py)：CookieCloud AES-CBC 解密组件，实现 PBKDF2 密钥派生 + AES 解密
- [dify_client.py](./components\py\dify_client.py)：Dify Webhook 推送客户端，自动从环境变量读取 api_token_dify 鉴权
- [env_loader.py](./components\py\env_loader.py)：环境变量加载器，从脚本目录 .env 读取配置（青龙面板优先），兼容 python-dotenv 缺失场景
- [html_parser.py](./components\py\html_parser.py)：HTML转纯文本清洗工具，去除标签、空行、多余换行；支持可选移除 script/style
- [pg_client.py](./components\py\pg_client.py)：PostgreSQL通用连接池组件，通过环境变量注入配置，提供查询df方法
- [redis_client.py](./components\py\redis_client.py)：Redis通用连接客户端，通过环境变量注入配置，提供连接实例与基础list操作
- [text_dedup.py](./components\py\text_dedup.py)：基于 SimHash + 编辑距离的文本去重组件，适用于新闻标题/短文本去重
