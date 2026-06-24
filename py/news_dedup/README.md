# 新闻资讯去重推送 news_dedup

## 基础信息
- 脚本入口：`py/news_dedup/main.py`
- 开发语言：Python3
- 依赖公共组件：
  1. `components/py/text_dedup.py` SimHash 文本去重
  2. `components/py/dify_client.py` Dify Webhook 推送
- 青龙推荐定时：0 7 * * *

## 功能简介
1. 从 Miniflux 查询多 feed 新闻资讯；
2. 使用 SimHash + 编辑距离双重去重；
3. 仅推送去重后数据至 Dify Webhook。

## 运行产出
无文件产出

## 环境变量配置
复制 config.tpl.env 内变量填入青龙面板环境变量；

## 本地调试运行
```bash
cd py/news_dedup
python3 main.py
```

## 更新记录
2026-06-24：脚本初始化