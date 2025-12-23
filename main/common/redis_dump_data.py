# coding=utf8
import os,redis,json
import pandas as pd

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

topic_data = redis_conn.lrange('hr_topic_train_data_0', 0, -1)

encoded_topic_data = [json.loads(item) for item in topic_data]
title_list = [item["title"] for item in encoded_topic_data]

df = pd.DataFrame(title_list, columns=["title"])

df.to_csv("/mnt/score.csv",index=False, encoding='utf-8-sig')

print("数据已导出")