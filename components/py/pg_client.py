#!/usr/bin/env python3
# @desc PostgreSQL通用连接池组件，通过环境变量注入配置，提供查询df方法
# coding=utf8
import os
import pandas as pd
from sqlalchemy import create_engine, text

def get_pg_engine(prefix: str = "fetch_feed_"):
    """
    获取PG连接引擎
    :param prefix: 环境变量前缀，区分多套数据库
    """
    db_config = {
        "host": os.getenv(f"{prefix}pghost"),
        "port": int(os.getenv(f"{prefix}pgport", 5432)),
        "dbname": os.getenv(f"{prefix}pgdb"),
        "user": os.getenv(f"{prefix}pguser", "postgres"),
        "password": os.getenv(f"{prefix}pgpwd"),
    }
    engine = create_engine(
        "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(**db_config),
        pool_pre_ping=True
    )
    return engine

def query_to_df(engine, sql_text: text, params: dict):
    """执行SQL，返回DataFrame"""
    return pd.read_sql(sql_text, con=engine, params=params)

if __name__ == "__main__":
    # 组件单独调试入口
    e = get_pg_engine()
    print("PG连接引擎创建成功")