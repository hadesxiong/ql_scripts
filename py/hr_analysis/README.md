# 人力资源自媒体资讯分析 hr_analysis

## 描述
从 Miniflux 拉取 HR 资讯清洗后推送 Dify

## 基础信息
- 脚本入口：`py/hr_analysis/main.py`
- 开发语言：Python3
- 依赖公共组件：
  1. `components/py/html_parser.py` HTML 正文清洗
  2. `components/py/dify_client.py` Dify Webhook 推送
- 青龙推荐定时：0 8 * * *

## 功能简介
1. 从 Miniflux 查询当日人力资源相关资讯；
2. 清洗 HTML 正文，提取纯文本；
3. 推送全量数据至 Dify Webhook 做进一步分析。

## 运行产出
无文件产出

## 环境变量配置
复制 config.tpl.env 内变量填入青龙面板环境变量；

## 本地调试运行
```bash
cd py/hr_analysis
python3 main.py
```

## 更新记录
2026-06-24：脚本初始化