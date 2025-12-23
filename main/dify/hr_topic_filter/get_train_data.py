# coding=utf8
# 获取一定数量的数据
import os, redis, re, json
import pandas as pd

from datetime import datetime, timedelta, time
from sqlalchemy import create_engine, text

# 配置postgresql的参数

DB_CONFIG = {
    "host": os.getenv("PGHOST", "postgresql_sit1"),
    "port": int(os.getenv("PGPORT", 5432)),
    "dbname": os.getenv("PGDATABASE", "app_freshrss_db"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD"),
}

# 配置时间

today_utc = datetime.today()
START_TS = int(
    datetime.combine(today_utc - timedelta(days=30), time.min).timestamp())
END_TS = int(datetime.combine(today_utc, time.max).timestamp())

# 建立数据库连接

engine = create_engine(
    "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(
        **DB_CONFIG),
    pool_pre_ping=True)

# 拼装语句
sql = text(f"""
    SELECT id, title
    FROM "Nero_entry"
    WHERE date BETWEEN :start_ts AND :end_ts
""")

# 形成数据矩阵
df = pd.read_sql(sql,
                 con=engine,
                 params={
                     "start_ts": START_TS,
                     "end_ts": END_TS
                 })
# 转换成dict
data_list = df.to_dict(orient="records")
print(len(data_list))

# 筛选可能相关的话题
hr_patterns = [
    r"(工资|薪资|薪酬|奖金|年终奖|福利|社保|公积金)", r"(晋升|升职|加薪|跳槽|离职|裁员|招聘|面试)",
    r"(劳动法|劳动合同|劳动仲裁|人社部|政策|规定|条例)"
]

samples = []

for each in data_list:
    for pattern in hr_patterns:
        if re.search(pattern, each["title"]):
            samples.append({
                "title": each["title"],
                "label": "人事相关",
                "category": "待细分"
            })

print(len(samples))
# 将数据写入redis，每1000条写一轮

REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "redis_sit1"),
    "port": os.getenv("REDIS_PORT", 6379),
    "password": os.getenv("REDIS_PWD"),
    "db": os.getenv("REDIS_DB", 0),
    "decode_responses": os.getenv("REDIS_DECODE", "true"),
    "encoding": os.getenv("REDIS_ENCODE", "utf-8")
}

redis_url = "redis://:{password}@{host}:{port}/{db}?decode_responses={decode_responses}&encoding={encoding}".format(
    **REDIS_CONFIG)

redis_pool = redis.ConnectionPool.from_url(redis_url)

redis_conn = redis.Redis(connection_pool=redis_pool)

for group_num in range(0, len(samples), 1000):

    chunk = samples[group_num:group_num + 1000]
    title = f"hr_topic_train_data_{group_num}"

    for index, item in enumerate(chunk):
        redis_conn.rpush(title, json.dumps(item))
