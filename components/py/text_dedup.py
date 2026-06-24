#!/usr/bin/env python3
# @desc 基于 SimHash + 编辑距离的文本去重组件，适用于新闻标题/短文本去重
# coding=utf8
import time
import simhash
import jieba
from typing import List, Dict
from Levenshtein import ratio

class TextDedup:
    STOP_WORDS = {"的", "了", "是", "在", "宣布", "据悉", "将", "目前", "已", "共", "约"}
    HAMMING_THRESHOLD = 8
    EDIT_RATIO_THRESHOLD = 0.9

    def __init__(self, hash_bits: int = 64):
        self.hash_bits = hash_bits
        self._map: Dict[str, tuple[int, str]] = {}

    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        for ch in ["，", "。", " ", "、", "：", "：", "！", "？", "【", "】", "（", "）"]:
            text = text.replace(ch, "")
        words = jieba.lcut(text.strip())
        valid = [w for w in words if w not in self.STOP_WORDS and len(w) >= 1]
        return "".join(valid)

    def _simhash(self, clean_text: str) -> int:
        if not clean_text:
            return 0
        return simhash.Simhash(clean_text, f=self.hash_bits).value

    @staticmethod
    def _hamming(h1: int, h2: int) -> int:
        return bin(h1 ^ h2).count("1")

    def is_dup(self, item_id: str, title: str) -> bool:
        clean = self._normalize(title)
        cur_hash = self._simhash(clean)
        for eid, (ehash, eclean) in self._map.items():
            if self._hamming(cur_hash, ehash) <= self.HAMMING_THRESHOLD or \
               ratio(clean, eclean) >= self.EDIT_RATIO_THRESHOLD:
                return True
        self._map[item_id] = (cur_hash, clean)
        return False

    def batch_check(self, items: List[Dict]) -> List[bool]:
        start = time.time()
        res = [self.is_dup(item["url"], item["title"]) for item in items]
        cost = time.time() - start
        print(f"批量 {len(items)} 条 | 耗时 {cost:.2f}s | 平均 {cost/len(items)*1000:.2f}ms/条")
        return res