# 青龙私有脚本仓库
青龙挂载路径：`/ql/scripts`
同步方式：定时执行 utils_sync.sh（git pull 硬覆盖）

## 环境变量说明
本仓库无顶层全局 env_sample 目录，配置分两级管理：
1. **单脚本专属变量**
每个业务脚本文件夹内自带 `config.tpl.env`，复制模板内键值填入青龙环境变量即可使用；
2. **全局公共组件变量**
推送、第三方接口等全局密钥，直接在青龙面板全局环境自行添加，无统一模板；
3. 安全规范
真实密钥、本地 .env 文件已加入 gitignore 禁止提交，仅允许仓库留存 `config.tpl.env` 模板。

## 快速文档入口
- 公共组件清单：[docs/components_list.md](./docs/components_list.md)
- 全脚本总表：[docs/script_info.md](./docs/script_info.md)
- 全局更新记录：[UPDATE_LOG.md](./UPDATE_LOG.md)

## 仓库部署教程
1. 青龙服务器克隆私有仓库至 /ql/scripts
2. 配置git免密拉取（SSH密钥/GitHub Personal Token）
3. 添加定时任务执行 utils_sync.sh 实现自动同步

<!-- AUTO_SCRIPT_LIST_START -->
## 全部业务脚本清单（自动生成）

### Python脚本
- [新闻资讯去重推送 news_dedup](./py/news_dedup/README.md)：1. 从 Miniflux 查询多 feed 新闻资讯；
- [每日工作日报 daily_report](./py/daily_report/README.md)：1. 查询 Plane 数据库中 7 日内待完成任务；
- [CookieCloud-Cookie同步脚本 async_cookie](./py/async_cookie/README.md)：1. 从 CookieCloud 服务器拉取加密 Cookie 数据；
- [每周工作周报 weekly_report](./py/weekly_report/README.md)：1. 查询 Plane 数据库中本周已完成任务；
- [资讯Feed同步脚本 fetch_feed](./py/fetch_feed/README.md)：1. 读取PostgreSQL中昨日资讯数据；
- [人力资源自媒体资讯分析 hr_analysis](./py/hr_analysis/README.md)：1. 从 Miniflux 查询当日人力资源相关资讯；

<!-- AUTO_SCRIPT_LIST_END -->