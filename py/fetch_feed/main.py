#!/usr/bin/env python3
# @name 资讯Feed同步脚本
# @desc 读取PostgreSQL昨日资讯，清洗HTML正文后存入Redis字符串
# @output output/fetch_feed/run_log.csv
# coding=utf8
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta, time
from sqlalchemy import text

# 固定寻址：当前脚本向上两层 = 仓库根目录，导入components
BASE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_ROOT))

# 导入公共组件
from components.py.pg_client import get_pg_engine, query_to_df
from components.py.html_parser import html_to_text
from components.py.redis_client import get_redis_conn

# ====================== 业务常量配置 ======================
TABLE_NAME = os.getenv("fetch_feed_table_name", "Nero_entry")
TS_COL = "date"
ID_COL = "id_feed"
ID_FEEDS = [76]

# 计算昨日时间戳
today_utc = datetime.today()
start_day = today_utc - timedelta(days=1)
START_TS = int(datetime.combine(start_day, time.min).timestamp())
END_TS = int(datetime.combine(start_day, time.max).timestamp())
redis_key = f'{start_day.strftime("%Y-%m-%d")}_hr_topic'
# ========================================================

# 初始化redis_conn
redis_conn = get_redis_conn()

def main():
    # 1. 初始化PG连接并查询数据
    engine = get_pg_engine(prefix="fetch_feed_")
    sql = text(f"""
        SELECT title, link, id_feed, tags, content
        FROM "{TABLE_NAME}"
        WHERE {TS_COL} BETWEEN :start_ts AND :end_ts
        AND {ID_COL} = ANY(:id_feeds)
    """)
    df = query_to_df(
        engine=engine,
        sql_text=sql,
        params={
            "start_ts": START_TS,
            "end_ts": END_TS,
            "id_feeds": ID_FEEDS
        }
    )
    print(f"共抽取 {len(df)} 条资讯记录")

    # 2. HTML正文清洗
    df['content_plain'] = df['content'].apply(html_to_text)
    records = df.to_dict(orient="records")

    # 3. 写入Redis
    record_str = json.dumps(records, ensure_ascii=False, indent=None)
    redis_conn.set(redis_key, record_str)
    print(f"数据写入Redis完成，key={redis_key}")

    # 可选：落地运行日志到统一output目录
    log_path = BASE_ROOT / "output" / "fetch_feed" / "run_log.csv"
    df.to_csv(log_path, index=False, encoding="utf-8-sig")
    print(f"本地运行日志已保存至 {log_path}")

if __name__ == "__main__":
    main()