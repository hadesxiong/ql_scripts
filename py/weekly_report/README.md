# 每周工作周报 weekly_report

## 描述
查询 Plane 本周完成工作并汇总推送

## 基础信息
- 脚本入口：`py/weekly_report/main.py`
- 开发语言：Python3
- 依赖公共组件：无
- 青龙推荐定时：0 10 * * 1

## 功能简介
1. 查询 Plane 数据库中本周已完成任务；
2. 逐条推送任务详情至 Hookshot；
3. 汇总本周工作量点数。

## 运行产出
无文件产出

## 环境变量配置
复制 config.tpl.env 内变量填入青龙面板环境变量；

## 本地调试运行
```bash
cd py/weekly_report
python3 main.py
```

## 更新记录
2026-06-24：脚本初始化