#!/usr/bin/env python3
"""
彩票分析系统回测框架

提供滚动回测（walk-forward backtest）能力，用于量化评估 3D/排三、大乐透
推荐策略的历史表现。仅使用标准库。
"""

import math
import random
from collections import Counter
from typing import List, Dict, Callable, Tuple, Optional

# ═══════════════════════════════════════════════════════════════
#  奖金近似表（单位：元）
# ═══════════════════════════════════════════════════════════════

# 3D/排三奖金（单注2元）
PRIZE_3D_DIRECT = 1040          # 直选
PRIZE_3D_GROUP3 = 346           # 组三
PRIZE_3D_GROUP6 = 173           # 组六
COST_PER_BET_3D = 2

# 大乐透奖金近似表（单注2元，追加忽略）
# 奖级 -> 近似奖金
PRIZE_DLT = {
    (5, 2): 5_000_000,   # 一等奖（浮动，忽略精确值）
    (5, 1): 100_000,     # 二等奖近似
    (5, 0): 10_000,      # 三等奖
    (4, 2): 3_000,       # 四等奖
    (4, 1): 300,         # 五等奖
    (4, 0): 200,         # 六等奖
    (3, 2): 200,         # 六等奖
    (3, 1): 15,          # 八等奖
    (3, 0): 5,           # 九等奖
    (2, 2): 15,          # 八等奖
    (2, 1): 5,           # 九等奖
    (1, 2): 5,           # 九等奖
    (0, 2): 5,           # 九等奖
}
COST_PER_BET_DLT = 2


def _shape_type(digits: Tuple[int, int, int]) -> str:
    """判断 3D 号码形态：豹子/组三/组六。"""
    unique = len(set(digits))
    if unique == 1:
        return 'baozi'
    elif unique == 2:
        return 'zusan'
    return 'zuliu'


def _match_3d_group(actual: Tuple[int, int, int], predicted: Tuple[int, int, int]) -> bool:
    """组选命中：两个号码数字集合相同。"""
    return set(actual) == set(predicted)


def _dlt_prize_level(front_hits: int, back_hits: int) -> Optional[Tuple[int, int]]:
    """返回 (前区命中, 后区命中)，若未中奖返回 None。"""
    if front_hits == 5 and back_hits == 2:
        return (5, 2)
    if front_hits == 5 and back_hits == 1:
        return (5, 1)
    if front_hits == 5 and back_hits == 0:
        return (5, 0)
    if front_hits == 4 and back_hits == 2:
        return (4, 2)
    if front_hits == 4 and back_hits == 1:
        return (4, 1)
    if front_hits == 4 and back_hits == 0:
        return (4, 0)
    if front_hits == 3 and back_hits == 2:
        return (3, 2)
    if front_hits == 3 and back_hits == 1:
        return (3, 1)
    if front_hits == 3 and back_hits == 0:
        return (3, 0)
    if front_hits == 2 and back_hits == 2:
        return (2, 2)
    if front_hits == 2 and back_hits == 1:
        return (2, 1)
    if front_hits == 1 and back_hits == 2:
        return (1, 2)
    if front_hits == 0 and back_hits == 2:
        return (0, 2)
    return None


def backtest_3d(
    data: List[Dict],
    recommender_fn: Callable[[List[Dict]], Dict],
    train_window: int = 100,
    test_window: int = 10,
    step: int = 1
) -> Dict:
    """
    滚动回测 3D/排三推荐策略。

    数据从新到旧排列（data[0] 为最新），对每一个测试点 t，使用
    data[t+1 : t+train_window+1] 作为训练集（t 之前的 train_window 期），
    预测 data[t] 等接下来若干期。

    统计指标：
    - 直选命中次数/率
    - 组选命中次数/率（豹子按直选计）
    - 平均命中位数（位置相同数字个数）
    - ROI（按单注2元，直选1040元，组三346元，组六173元）

    Args:
        data: 开奖数据列表，每项包含 h, t, u 字段。
        recommender_fn: 推荐函数，接收训练数据，返回包含 'bets' 的 Dict。
            'bets' 为 List[Tuple[int, int, int]] 或 List[Dict]（含 b/s/g 字段）。
        train_window: 训练窗口大小。
        test_window: 每次预测的期数。
        step: 滚动步长。

    Returns:
        回测统计结果字典。
    """
    n = len(data)
    if n < train_window + test_window:
        return {
            'error': '数据量不足以完成回测',
            'data_count': n,
            'train_window': train_window,
            'test_window': test_window,
        }

    total_tests = 0
    direct_hits = 0
    group_hits = 0
    total_digit_hits = 0.0
    total_cost = 0
    total_prize = 0

    # 兼容 recommender_fn 输出格式
    def _extract_bets(rec: Dict) -> List[Tuple[int, int, int]]:
        bets = rec.get('bets', [])
        result = []
        for b in bets:
            if isinstance(b, dict):
                result.append((int(b.get('b', b.get('h', 0))),
                               int(b.get('s', b.get('t', 0))),
                               int(b.get('g', b.get('u', 0)))))
            elif isinstance(b, (list, tuple)) and len(b) >= 3:
                result.append((int(b[0]), int(b[1]), int(b[2])))
        return result

    max_start = n - train_window - test_window
    for start in range(0, max_start + 1, step):
        train_start = start + 1
        train_end = start + train_window + 1
        train_data = data[train_start:train_end]
        if len(train_data) < train_window:
            continue

        rec = recommender_fn(train_data)
        bets = _extract_bets(rec)
        total_cost += len(bets) * COST_PER_BET_3D

        for offset in range(test_window):
            test_idx = start + offset
            if test_idx >= n:
                break
            actual = (data[test_idx]['h'], data[test_idx]['t'], data[test_idx]['u'])
            total_tests += 1

            best_digit_hits = 0
            hit_direct = False
            hit_group = False

            for bet in bets:
                digit_hits = sum(1 for a, p in zip(actual, bet) if a == p)
                best_digit_hits = max(best_digit_hits, digit_hits)
                if actual == bet:
                    hit_direct = True
                if _match_3d_group(actual, bet):
                    hit_group = True

            total_digit_hits += best_digit_hits

            if hit_direct:
                direct_hits += 1
                total_prize += PRIZE_3D_DIRECT
            elif hit_group:
                group_hits += 1
                shape = _shape_type(actual)
                if shape == 'zusan':
                    total_prize += PRIZE_3D_GROUP3
                elif shape == 'zuliu':
                    total_prize += PRIZE_3D_GROUP6

    roi = (total_prize - total_cost) / total_cost if total_cost > 0 else 0.0

    return {
        'game': '3d',
        'total_tests': total_tests,
        'train_window': train_window,
        'test_window': test_window,
        'step': step,
        'bets_per_test': len(_extract_bets(recommender_fn(data[1:1 + train_window]))) if n > train_window else 0,
        'direct_hits': direct_hits,
        'direct_hit_rate': round(direct_hits / total_tests, 6) if total_tests > 0 else 0.0,
        'group_hits': group_hits,
        'group_hit_rate': round(group_hits / total_tests, 6) if total_tests > 0 else 0.0,
        'avg_digit_hits': round(total_digit_hits / total_tests, 4) if total_tests > 0 else 0.0,
        'total_cost': total_cost,
        'total_prize': total_prize,
        'net_profit': total_prize - total_cost,
        'roi': round(roi, 6),
    }


def backtest_dlt(
    data: List[Dict],
    recommender_fn: Callable[[List[Dict]], Dict],
    train_window: int = 100,
    test_window: int = 10,
    step: int = 1
) -> Dict:
    """
    滚动回测大乐透推荐策略。

    数据从新到旧排列（data[0] 为最新），对每一个测试点 t，使用
    data[t+1 : t+train_window+1] 作为训练集，预测 data[t] 等接下来若干期。

    统计指标：
    - 前区命中数分布
    - 后区命中数分布
    - 奖级命中分布（5+2, 5+1, 5+0, 4+2, ...）
    - ROI（按单注2元，奖金近似表计算）

    Args:
        data: 开奖数据列表，每项包含 front, back 字段。
        recommender_fn: 推荐函数，接收训练数据，返回包含 'bets' 的 Dict。
            'bets' 为 List[Dict]，每个 Dict 包含 'front'(List[int]) 和 'back'(List[int])。
        train_window: 训练窗口大小。
        test_window: 每次预测的期数。
        step: 滚动步长。

    Returns:
        回测统计结果字典。
    """
    n = len(data)
    if n < train_window + test_window:
        return {
            'error': '数据量不足以完成回测',
            'data_count': n,
            'train_window': train_window,
            'test_window': test_window,
        }

    total_tests = 0
    front_hit_dist: Counter = Counter()
    back_hit_dist: Counter = Counter()
    level_dist: Counter = Counter()
    total_cost = 0
    total_prize = 0

    def _extract_bets(rec: Dict) -> List[Dict]:
        bets = rec.get('bets', [])
        result = []
        for b in bets:
            if isinstance(b, dict):
                front = list(b.get('front', []))
                back = list(b.get('back', []))
                if len(front) == 5 and len(back) == 2:
                    result.append({'front': front, 'back': back})
            elif isinstance(b, (list, tuple)) and len(b) >= 2:
                front = list(b[0]) if isinstance(b[0], (list, tuple)) else []
                back = list(b[1]) if isinstance(b[1], (list, tuple)) else []
                if len(front) == 5 and len(back) == 2:
                    result.append({'front': front, 'back': back})
        return result

    max_start = n - train_window - test_window
    for start in range(0, max_start + 1, step):
        train_start = start + 1
        train_end = start + train_window + 1
        train_data = data[train_start:train_end]
        if len(train_data) < train_window:
            continue

        rec = recommender_fn(train_data)
        bets = _extract_bets(rec)
        total_cost += len(bets) * COST_PER_BET_DLT

        for offset in range(test_window):
            test_idx = start + offset
            if test_idx >= n:
                break
            actual = data[test_idx]
            afront = set(actual['front'])
            aback = set(actual['back'])
            total_tests += 1

            best_front = 0
            best_back = 0
            best_level = None

            for bet in bets:
                fh = len(set(bet['front']) & afront)
                bh = len(set(bet['back']) & aback)
                best_front = max(best_front, fh)
                best_back = max(best_back, bh)
                level = _dlt_prize_level(fh, bh)
                if level and (best_level is None or PRIZE_DLT.get(level, 0) > PRIZE_DLT.get(best_level, 0)):
                    best_level = level

            front_hit_dist[best_front] += 1
            back_hit_dist[best_back] += 1
            if best_level:
                level_dist[str(best_level)] += 1
                total_prize += PRIZE_DLT.get(best_level, 0)

    roi = (total_prize - total_cost) / total_cost if total_cost > 0 else 0.0

    return {
        'game': 'dlt',
        'total_tests': total_tests,
        'train_window': train_window,
        'test_window': test_window,
        'step': step,
        'bets_per_test': len(_extract_bets(recommender_fn(data[1:1 + train_window]))) if n > train_window else 0,
        'front_hit_distribution': dict(front_hit_dist),
        'back_hit_distribution': dict(back_hit_dist),
        'level_distribution': dict(level_dist),
        'total_cost': total_cost,
        'total_prize': total_prize,
        'net_profit': total_prize - total_cost,
        'roi': round(roi, 6),
    }


def _dummy_3d_recommender(data: List[Dict]) -> Dict:
    """3D 示例推荐函数：随机生成10注组六。仅用于测试。"""
    bets = []
    seen = set()
    while len(bets) < 10:
        combo = tuple(sorted(random.sample(range(10), 3)))
        if combo not in seen:
            seen.add(combo)
            bets.append(combo)
    return {'bets': bets}


def _dummy_dlt_recommender(data: List[Dict]) -> Dict:
    """大乐透示例推荐函数：随机生成5注。仅用于测试。"""
    bets = []
    for _ in range(5):
        front = sorted(random.sample(range(1, 36), 5))
        back = sorted(random.sample(range(1, 13), 2))
        bets.append({'front': front, 'back': back})
    return {'bets': bets}


if __name__ == '__main__':
    # 简单自测：构造伪造数据
    fake_3d = [{'h': (i * 3) % 10, 't': (i * 5) % 10, 'u': (i * 7) % 10} for i in range(130)]
    print('3d backtest:', backtest_3d(fake_3d, _dummy_3d_recommender, train_window=100, test_window=10, step=5))

    fake_dlt = [{'front': sorted(random.sample(range(1, 36), 5)),
                 'back': sorted(random.sample(range(1, 13), 2))} for _ in range(130)]
    print('dlt backtest:', backtest_dlt(fake_dlt, _dummy_dlt_recommender, train_window=100, test_window=10, step=5))
