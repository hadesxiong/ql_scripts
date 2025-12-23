# coding=utf8

import os, json, redis
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, time
from sqlalchemy import create_engine, text

# 1. 配置postgresql参数
DB_CONFIG = {
    "host": os.getenv("PGHOST", "postgresql_sit1"),
    "port": int(os.getenv("PGPORT", 5432)),
    "dbname": os.getenv("PGDATABASE", "app_freshrss_db"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Faurecia614"),
}

TABLE_NAME = "Nero_entry"  # 目标表
TS_COL = "date"  # int8 时间戳字段名
ID_COL = "id_feed"  # feed id 字段名

# 把自然日期转成秒级时间戳（UTC）
today_utc = datetime.today()
START_TS = int(
    datetime.combine(today_utc - timedelta(days=2), time.min).timestamp())
END_TS = int(datetime.combine(today_utc, time.max).timestamp())

ID_FEEDS = [3, 4, 5, 6, 9, 19, 28, 44, 50, 51, 52, 57, 76, 82,
            83]  # 目标 id_feed 列表

# 2. 建立postgresql数据库连接
engine = create_engine(
    "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(
        **DB_CONFIG),
    pool_pre_ping=True)

# 3. 拼装sql语句
sql = text(f"""
    SELECT title, link, id_feed, tags, content
    FROM "Nero_entry"
    WHERE {TS_COL} BETWEEN :start_ts AND :end_ts
    AND {ID_COL} = ANY(:id_feeds)
""")

df = pd.read_sql(sql,
                 con=engine,
                 params={
                     "start_ts": START_TS,
                     "end_ts": END_TS,
                     "id_feeds": ID_FEEDS
                 })

# 4. 打印postgresql查询结果
print(f"共抽取 {len(df)} 条记录")

# 利用beautifulsoup格式化content字段
def html_to_text(html_content):
    if pd.isna(html_content):
        return ''
    try:
        soup = BeautifulSoup(str(html_content), 'html.parser')
        text = soup.get_text(separator = '\n', strip = True)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        return text
    except Exception as e:
        print('解析错误: {e}')
        return str(html_content)

df['content_plain'] = df['content'].apply(html_to_text)
records = df.to_dict(orient="records")

# 5. 将数据写入redis
redis_host = os.getenv('base_redis_host')
redis_port = os.getenv('base_redis_port')
redis_pwd = os.getenv('base_redis_pwd')

redis_pool = redis.ConnectionPool(host=redis_host,
                                  port=redis_port,
                                  password=redis_pwd,
                                  db=0,
                                  decode_responses=True,
                                  encoding='utf-8')
redis_conn = redis.Redis(connection_pool=redis_pool)

record_str = json.dumps(records, ensure_ascii=False, indent=None)
redis_conn.set(f'{datetime.now().strftime("%Y-%m-%d")}_hr_topic', record_str)
