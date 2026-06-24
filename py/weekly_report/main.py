#!/usr/bin/env python3
# @name 每周工作周报
# @desc 查询 Plane 本周已完成任务，推送周报摘要到 Hookshot
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
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)

    engine = create_engine(
        "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(**DB_CONFIG),
        pool_pre_ping=True
    )

    projects = pd.read_sql("SELECT name, id FROM \"projects\"", con=engine)
    estimates = pd.read_sql("SELECT id, value FROM \"estimate_points\"", con=engine)

    sql = text("""
        SELECT name, project_id, estimate_point_id, completed_at
        FROM "issues"
        WHERE completed_at BETWEEN :start AND :end
    """)
    df = pd.read_sql(sql, con=engine, params={"start": week_start, "end": week_end})

    proj_map = projects.set_index('id')['name']
    est_map = estimates.set_index('id')['value']

    df['project_id'] = df['project_id'].map(proj_map)
    df['estimate_point_id'] = df['estimate_point_id'].map(est_map)
    df['completed_at'] = df['completed_at'].dt.strftime('%Y-%m-%d %H:%M')
    df = df.rename(columns={'project_id': 'project', 'estimate_point_id': 'estimate'})

    records = df.to_dict(orient='records')

    webhook_url = os.getenv('weekly_report_webhook_url')
    if not webhook_url:
        print("❌ 缺少 weekly_report_webhook_url")
        return

    headers = {"Content-Type": "application/json"}
    time.sleep(2)
    requests.post(webhook_url, json={"text": f"--- 通报日期: {today} ---"}, headers=headers)
    time.sleep(2)

    if not records:
        requests.post(webhook_url, json={"text": "本周还未有完成的工作事项。"}, headers=headers)
    else:
        for r in records:
            msg = f"名称: {r['project']}-{r['name']}, 完成日期: {r['completed_at']}, 工作量: {r['estimate']}"
            requests.post(webhook_url, json={"text": msg}, headers=headers)
            time.sleep(2)

        df['estimate'] = pd.to_numeric(df['estimate'], errors='coerce')
        total = df['estimate'].sum()
        requests.post(webhook_url, json={"text": f"截止至今，本周共完成工作事项{len(records)}项, 工作量总计{total}点"}, headers=headers)
        time.sleep(2)

    requests.post(webhook_url, json={"text": "--- 通报完毕 ---"}, headers=headers)


if __name__ == "__main__":
    main()