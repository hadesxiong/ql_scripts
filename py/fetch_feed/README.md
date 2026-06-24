# 资讯Feed同步脚本 fetch_feed

## 基础信息
- 脚本入口：`py/fetch_feed/main.py`
- 开发语言：Python3
- 依赖公共组件：
  1. `components/py/pg_client.py` PostgreSQL数据库连接封装
  2. `components/py/html_parser.py` HTML正文清洗工具
  3. `components/py/redis_client.py` Redis数据库连接封装
- 青龙推荐定时：0 1 * * *

## 功能简介
1. 读取PostgreSQL中昨日资讯数据；
2. 清洗content字段HTML标签，生成纯文本content_plain；
3. 全量资讯列表JSON序列化存入Redis，key为`{日期}_hr_topic`；
4. 本地落地csv运行日志至output/fetch_feed/。

## 运行产出
1. Redis：`YYYY-MM-DD_hr_topic` 字符串存储全量资讯
2. 文件产出：`output/fetch_feed/run_log.csv`

## 环境变量配置
复制 config.tpl.env 内变量填入青龙面板环境变量；
Redis相关变量为全局公共配置，无需重复填写。

## 本地调试运行
```bash
cd py/fetch_feed
python3 main.py
```
## 更新记录
2026-06-24：脚本初始化，完成 PG 查询、HTML 清洗、Redis 写入全流程