#!/usr/bin/env python3
# @desc Redis通用连接客户端，通过环境变量注入配置，提供连接实例与基础list操作
# coding=utf8
import os
import redis
import json

def get_redis_conn(prefix: str = "") -> redis.Redis:
    """
    获取Redis连接实例
    :param prefix: 环境变量前缀，多Redis实例时区分，默认无前缀
    """
    config = {
        "host": os.getenv(f"{prefix}REDIS_HOST", "redis_sit1"),
        "port": int(os.getenv(f"{prefix}REDIS_PORT", 6379)),
        "password": os.getenv(f"{prefix}REDIS_PWD"),
        "db": int(os.getenv(f"{prefix}REDIS_DB", 0)),
        "decode_responses": os.getenv(f"{prefix}REDIS_DECODE", "true") == "true",
        "encoding": os.getenv(f"{prefix}REDIS_ENCODE", "utf-8")
    }
    url = "redis://:{password}@{host}:{port}/{db}?decode_responses={decode_responses}&encoding={encoding}".format(**config)
    pool = redis.ConnectionPool.from_url(url)
    return redis.Redis(connection_pool=pool)

def list_get_all(rd: redis.Redis, key: str):
    """读取list全部数据并json反序列化"""
    raw_list = rd.lrange(key, 0, -1)
    return [json.loads(item) for item in raw_list]

def str_set_json(rd: redis.Redis, key: str, data, ensure_ascii=False):
    """字典/列表存入string，自动json序列化"""
    json_str = json.dumps(data, ensure_ascii=ensure_ascii, indent=None)
    rd.set(key, json_str)

if __name__ == "__main__":
    # 组件自测入口
    rd_client = get_redis_conn()
    print("Redis连接创建成功")