#!/usr/bin/env python3
# @name 新闻资讯去重推送
# @desc 从 Miniflux 拉取多 feed 资讯，SimHash 去重后推送 Dify Webhook
# coding=utf8
import os
import sys
import pytz
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, time, timezone
from sqlalchemy import create_engine, text

BASE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_ROOT))

from components.py.text_dedup import TextDedup
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
    ID_FEEDS = [13, 15, 21, 23]

    tz = pytz.timezone('Asia/Shanghai')
    today = datetime.now(tz).date()
    yesterday = today - timedelta(days=1)

    start = tz.localize(datetime(yesterday.year, yesterday.month, yesterday.day, 20, 0, 0))
    end = tz.localize(datetime(today.year, today.month, today.day, 19, 59, 59, 999999))

    engine = create_engine(
        "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(**DB_CONFIG),
        pool_pre_ping=True
    )

    sql = text(f"""
        SELECT title, url FROM "{TABLE_NAME}"
        WHERE {TS_COL} BETWEEN :start AND :end
        AND {ID_COL} = ANY(:feeds)
    """)
    df = pd.read_sql(sql, con=engine, params={"start": start.astimezone(timezone.utc),
                                               "end": end.astimezone(timezone.utc),
                                               "feeds": ID_FEEDS})

    records = df.to_dict(orient='records') if not df.empty else []

    dedup = TextDedup()
    result = dedup.batch_check(records)
    filtered = [item for item, flag in zip(records, result) if not flag]

    token = os.getenv('news_dedup_webhook_dify')
    if not token:
        print("❌ 缺少 news_dedup_webhook_dify")
        return

    dify_webhook_send(token, {
        'data': filtered,
        'length': len(filtered),
        'date': today.strftime('%Y-%m-%d'),
        'exec_time': datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == "__main__":
    main()