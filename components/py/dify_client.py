#!/usr/bin/env python3
# @desc Dify Webhook 推送客户端，自动从环境变量读取 api_token_dify 鉴权
# coding=utf8
import os
import requests


def dify_webhook_send(webhook_token: str, data: dict, endpoint: str = "webhook"):
    """
    向 Dify Webhook 发送 JSON 数据
    :param webhook_token: webhook 路径令牌
    :param data: 请求体字典
    :param endpoint: "webhook" 正式 / "webhook-debug" 调试
    """
    url = f"http://app_dify-api-1:5001/triggers/{endpoint}/{webhook_token}"
    headers = {
        "Authorization": f"Bearer {os.getenv('api_token_dify')}",
        "Content-Type": "application/json"
    }
    resp = requests.post(url, json=data, headers=headers)
    print(resp.text)
    resp.raise_for_status()
    return resp