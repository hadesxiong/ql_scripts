#!/usr/bin/env python3
# @name 每日工作日报
# @desc 查询 Plane 中 7 日内未完成任务，推送到 Hookshot 通知
# coding=utf8
import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from sqlalchemy import create_engine, text

BASE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_ROOT))

from components.py.env_loader import load_script_env
load_script_env(Path(__file__).parent)

def main():
    DB_CONFIG = {
        'host': os.getenv('PGHOST', 'db_postgres_app'),
        'port': int(os.getenv('PGPORT', '5432')),
        'dbname': os.getenv('PGDATABASE', 'app_plane'),
        'user': os.getenv('pg_user'),
        'password': os.getenv('pg_password'),
    }

    today = date.today()
    end_date = today + timedelta(days=7)

    engine = create_engine(
        "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(**DB_CONFIG),
        pool_pre_ping=True
    )

    projects = pd.read_sql("SELECT name, id FROM \"projects\"", con=engine)
    estimates = pd.read_sql("SELECT id, value FROM \"estimate_points\"", con=engine)

    sql = text("""
        SELECT name, project_id, estimate_point_id, target_date
        FROM "issues"
        WHERE target_date BETWEEN :start AND :end
        AND completed_at IS NULL
    """)
    df = pd.read_sql(sql, con=engine, params={"start": today, "end": end_date})

    proj_map = projects.set_index('id')['name']
    est_map = estimates.set_index('id')['value']

    df['project_id'] = df['project_id'].map(proj_map)
    df['estimate_point_id'] = df['estimate_point_id'].map(est_map)
    df['target_date'] = df['target_date'].astype(str)
    df = df.rename(columns={'project_id': 'project', 'estimate_point_id': 'estimate'})

    records = df.to_dict(orient='records')

    webhook_url = os.getenv('daily_report_webhook_url')
    if not webhook_url:
        print("❌ 缺少 daily_report_webhook_url")
        return

    headers = {"Content-Type": "application/json"}
    time.sleep(2)
    requests.post(webhook_url, json={"text": f"--- 通报日期: {today} ---"}, headers=headers)
    time.sleep(2)

    if not records:
        requests.post(webhook_url, json={"text": "目前无7日内需要完成的工作事项, 好好享受片刻的宁静吧"}, headers=headers)
    else:
        for r in records:
            msg = f"名称: {r['project']}-{r['name']}, 目标完成日期: {r['target_date']}, 工作量估算: {r['estimate']}"
            requests.post(webhook_url, json={"text": msg}, headers=headers)
            time.sleep(2)

    requests.post(webhook_url, json={"text": "--- 通报完毕 ---"}, headers=headers)

if __name__ == "__main__":
    main()