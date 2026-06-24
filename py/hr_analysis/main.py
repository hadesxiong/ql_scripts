#!/usr/bin/env python3
# @name 人力资源自媒体资讯分析
# @desc 从 Miniflux 拉取当日人力资源资讯，清洗后推送 Dify Webhook
# coding=utf8
import os
import sys
import pytz
import pandas as pd
from pathlib import Path
from datetime import datetime, time, timezone
from sqlalchemy import create_engine, text

BASE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_ROOT))

from components.py.html_parser import html_to_text
from components.py.dify_client import dify_webhook_send

def main():
    DB_CONFIG = {
        'host': os.getenv('PGHOST', 'db_postgres_app'),
        'port': int(os.getenv('PGPORT', '5432')),
        'dbname': os.getenv('PGDATABASE', 'app_miniflux'),
        'user': os.getenv('pg_user'),
        'password': os.getenv('pg_password'),
    }

    TABLE_NAME = 'entries'
    TS_COL = 'published_at'
    ID_COL = 'feed_id'
    ID_FEEDS = [6]

    tz = pytz.timezone('Asia/Shanghai')
    today = datetime.now(tz).date()
    start = tz.localize(datetime(today.year, today.month, today.day, 0, 0, 0))
    end = tz.localize(datetime(today.year, today.month, today.day, 23, 59, 59, 999999))

    engine = create_engine(
        "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(**DB_CONFIG),
        pool_pre_ping=True
    )

    sql = text(f"""
        SELECT title, url, content, published_at, created_at
        FROM "{TABLE_NAME}"
        WHERE {TS_COL} BETWEEN :start AND :end
        AND {ID_COL} = ANY(:feeds)
    """)
    df = pd.read_sql(sql, con=engine, params={"start": start.astimezone(timezone.utc),
                                               "end": end.astimezone(timezone.utc),
                                               "feeds": ID_FEEDS})

    if not df.empty:
        df['published_at'] = df['published_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['created_at'] = df['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['content'] = df['content'].apply(lambda x: html_to_text(x, strip_special_tags=True, separator=' '))
        records = df.to_dict(orient='records')
    else:
        records = []

    token = os.getenv('hr_analysis_webhook_dify')
    if not token:
        print("❌ 缺少 hr_analysis_webhook_dify")
        return

    dify_webhook_send(token, {
        'data': records,
        'length': len(records),
        'date': today.strftime('%Y-%m-%d'),
        'exec_time': datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == "__main__":
    main()