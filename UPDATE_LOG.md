# 仓库全局更新日志
格式：[日期] | Commit短哈希 | 变更文件 | 脚本名称&简介
建议提交前手工运行 python3 tools/gen_all.py 保持同步

## 2026-06-24 | 初始提交
- `py/async_cookie/main.py`：CookieCloud Cookie同步脚本 | 从 CookieCloud 拉取加密 Cookie，解密后写入 .env 文件
- `py/daily_report/main.py`：每日工作日报 | 查询 Plane 中 7 日内未完成任务，推送到 Hookshot 通知
- `py/fetch_feed/main.py`：资讯Feed同步脚本 | 读取PostgreSQL昨日资讯，清洗HTML正文后存入Redis字符串
- `py/hr_analysis/main.py`：人力资源自媒体资讯分析 | 从 Miniflux 拉取当日人力资源资讯，清洗后推送 Dify Webhook
- `py/news_dedup/main.py`：新闻资讯去重推送 | 从 Miniflux 拉取多 feed 资讯，SimHash 去重后推送 Dify Webhook
- `py/weekly_report/main.py`：每周工作周报 | 查询 Plane 本周已完成任务，推送周报摘要到 Hookshot
## 2026-06-25 | Commit:c5e1331
- `py/async_cookie/main.py`：CookieCloud Cookie同步脚本 | 从 CookieCloud 拉取加密 Cookie，解密后写入 .env 文件
- `py/daily_report/main.py`：每日工作日报 | 查询 Plane 中 7 日内未完成任务，推送到 Hookshot 通知
- `py/fetch_feed/main.py`：资讯Feed同步脚本 | 读取PostgreSQL昨日资讯，清洗HTML正文后存入Redis字符串
- `py/hr_analysis/main.py`：人力资源自媒体资讯分析 | 从 Miniflux 拉取当日人力资源资讯，清洗后推送 Dify Webhook
- `py/news_dedup/main.py`：新闻资讯去重推送 | 从 Miniflux 拉取多 feed 资讯，SimHash 去重后推送 Dify Webhook
- `py/weekly_report/main.py`：每周工作周报 | 查询 Plane 本周已完成任务，推送周报摘要到 Hookshot