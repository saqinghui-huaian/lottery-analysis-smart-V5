#!/usr/bin/env python3
"""
一体化分析脚本 — 数据获取 + 统计分析 + 推荐生成 + 格式化输出
=================================================================

解决的问题：context压缩后AI丢失skill指令，导致分析质量下降。
本脚本将所有分析逻辑和格式化自动化，AI只需执行脚本、原样输出结果。

Usage:
    python3 run_analysis.py --games 3d          # 只分析福彩3D
    python3 run_analysis.py --games 35          # 只分析排三
    python3 run_analysis.py --games 3d,35       # 3D+排三
    python3 run_analysis.py --games 85          # 大乐透
    python3 run_analysis.py --games 3d,35,85    # 全部分析

输出：完整的格式化文本，直接复制粘贴即可。

数学基础:
- 组六72%, 组三27%, 豹子1%
- 和值7-20占83.2%, 跨度4-7占56.4%
- 奇偶1:2或2:1各37.5%, 大小1:2或2:1各37.5%
- 遗漏期望E=9期, P(≥15)=20.6%, P(≥20)=12.2%
"""

import sys
import os
import json
import argparse
import random
import io
from collections import Counter
from datetime import datetime

# Force UTF-8 on Windows so Chinese output is not garbled when piped/redirected.
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

# Add skill scripts directory to path
SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)

from lottery_analysis import fetch_3d, fetch_pl3, fetch_dlt
from smart_selector import SmartNumberSelector

# V4 modules — graceful fallback if not available yet
try:
    from smart_selector_v4 import SmartNumberSelectorV4
except Exception:
    SmartNumberSelectorV4 = None
try:
    from dlt_analysis_v4 import DLTAnalyzerV4
except Exception:
    DLTAnalyzerV4 = None
try:
    from ensemble import EnsembleRecommender3D, EnsembleRecommenderDLT
except Exception:
    EnsembleRecommender3D = None
    EnsembleRecommenderDLT = None


def compute_analysis_context(data, selector=None):
    """
    计算完整分析上下文，用于生成所有分析逻辑bullet。
    如果传入 selector，复用其 compute_context 中的高级统计。
    """
    ctx = {}
    n = len(data)

    # 复用 selector 的 compute_context 获取高级统计
    if selector is None:
        selector = SmartNumberSelector()
    advanced = selector.compute_context(data)
    ctx['advanced'] = advanced

    # ── 各位频率（近20/50期）──
    ctx['freq20'] = advanced['freq20']
    ctx['freq50'] = advanced['freq50']

    # ── 遗漏值 ──
    ctx['miss'] = advanced['miss']

    # ── 和值统计 ──
    sums = [d['h'] + d['t'] + d['u'] for d in data[:50]]
    ctx['hz_avg'] = sum(sums) / len(sums) if sums else 13.5

    # ── 和尾遗漏 ──
    ctx['sum_tail_miss'] = advanced['sum_tail_miss']

    # ── 跨度统计 ──
    spans = [max(d['h'], d['t'], d['u']) - min(d['h'], d['t'], d['u']) for d in data[:50]]
    ctx['kd_avg'] = sum(spans) / len(spans) if spans else 4.5

    # ── 跨度遗漏 ──
    ctx['span_miss'] = advanced['span_miss']

    # ── 形态统计（近50期）──
    ctx['shape_cnt'] = Counter()
    for d in data[:50]:
        unique = len(set([d['h'], d['t'], d['u']]))
        if unique == 1:
            ctx['shape_cnt']['豹子'] += 1
        elif unique == 2:
            ctx['shape_cnt']['组三'] += 1
        else:
            ctx['shape_cnt']['组六'] += 1
    
    # ── 奇偶比（近20期）──
    ctx['odd_even_cnt'] = Counter()
    for d in data[:20]:
        odd = sum(1 for x in [d['h'], d['t'], d['u']] if x % 2 == 1)
        ctx['odd_even_cnt'][f'{odd}:{3-odd}'] += 1
    
    # ── 大小比（近20期）──
    ctx['big_small_cnt'] = Counter()
    for d in data[:20]:
        big = sum(1 for x in [d['h'], d['t'], d['u']] if x >= 5)
        ctx['big_small_cnt'][f'{big}:{3-big}'] += 1
    
    # ── 012路频率（近20期）──
    ctx['road20'] = [Counter(), Counter(), Counter()]
    for d in data[:20]:
        for pos, key in enumerate(['h', 't', 'u']):
            ctx['road20'][pos][d[key] % 3] += 1
    
    return ctx


def _compute_exact_probs():
    """枚举全部1000种 (b,s,g) 组合，计算和值/和尾/跨度的精确理论概率。

    （V3.1）用精确枚举替代此前的近似常量，避免遗漏期望计算偏倚。
    """
    sum_probs = Counter()
    tail_probs = Counter()
    span_probs = Counter()
    for b in range(10):
        for s in range(10):
            for g in range(10):
                hz = b + s + g
                sum_probs[hz] += 1
                tail_probs[hz % 10] += 1
                span_probs[max(b, s, g) - min(b, s, g)] += 1
    return (
        {k: v / 1000 for k, v in sum_probs.items()},
        {k: v / 1000 for k, v in tail_probs.items()},
        {k: v / 1000 for k, v in span_probs.items()},
    )


_SUM_PROBS, _TAIL_PROBS, _SPAN_PROBS = _compute_exact_probs()


def expected_miss(prob):
    """几何分布期望遗漏 E = 1/p - 1（期）。"""
    return 1.0 / prob - 1 if prob > 0 else float('inf')


def _weighted_pick(signals, cap=25):
    """
    加权随机采样：从信号列表中按权重随机选择，而不是固定取最强。
    极端遗漏(>cap期)降权，避免每次都是跨度0/和尾4霸榜。
    参考：zhuangjl945/3D的多维度加权评分 + SELMA的概率归一化。
    """
    if not signals:
        return None
    # 权重 = min(miss, cap) — 极端遗漏不再有压倒性优势
    weighted = [(s, min(m[1], cap)) for s, m in [(s, s) for s in signals]]
    weights = [min(s[1], cap) for s in signals]
    total = sum(weights)
    if total == 0:
        return random.choice(signals)
    # 加权随机选择
    r = random.uniform(0, total)
    cumulative = 0
    for sig, w in zip(signals, weights):
        cumulative += w
        if r <= cumulative:
            return sig
    return signals[-1]


def get_miss_signals(ctx, threshold=15):
    """获取所有位置的遗漏信号，支持加权随机采样"""
    pos_map = {'bai': 0, 'shi': 1, 'ge': 2}
    results = {}
    for pos_key, label in [('bai', '百位'), ('shi', '十位'), ('ge', '个位')]:
        pos = pos_map[pos_key]
        signals = []
        for digit in range(10):
            m = ctx['miss'][pos].get(digit, 0)
            if m >= threshold:
                ratio = m / 9.0
                if m >= 25:
                    strength = '极强信号'
                elif m >= 20:
                    strength = '强信号'
                elif m >= 15:
                    strength = '中等信号'
                else:
                    strength = '弱信号'
                signals.append((digit, m, strength, ratio))
        signals.sort(key=lambda x: -x[1])
        results[label] = signals
    return results


def get_moderate_miss_signals(ctx, low=10, high=25):
    """获取中等遗漏信号（10-25期），比极端遗漏更有分析价值"""
    pos_map = {'bai': 0, 'shi': 1, 'ge': 2}
    results = {}
    for pos_key, label in [('bai', '百位'), ('shi', '十位'), ('ge', '个位')]:
        pos = pos_map[pos_key]
        signals = []
        for digit in range(10):
            m = ctx['miss'][pos].get(digit, 0)
            if low <= m <= high:
                ratio = m / 9.0
                signals.append((digit, m, '回补窗口', ratio))
        signals.sort(key=lambda x: -x[1])
        results[label] = signals
    return results


def get_recent_recovered(ctx, data, lookback=10):
    """获取近期回补的数字（之前遗漏较长，近10期内出现）"""
    pos_map = {'bai': 0, 'shi': 1, 'ge': 2}
    results = {}
    for pos_key, label in [('bai', '百位'), ('shi', '十位'), ('ge', '个位')]:
        pos = pos_map[pos_key]
        recovered = []
        for digit in range(10):
            # 当前遗漏
            current_miss = ctx['miss'][pos].get(digit, 0)
            # 检查近lookback期是否出现过
            appeared_recently = False
            for i, d in enumerate(data[:lookback]):
                key = ['h', 't', 'u'][pos]
                if d[key] == digit:
                    appeared_recently = True
                    break
            # 如果近10期出现过，但之前有较长遗漏（当前miss=0说明刚出现）
            if appeared_recently and current_miss <= 3:
                # 计算这个数字在更早之前的遗漏
                for i, d in enumerate(data[lookback:], start=lookback):
                    key = ['h', 't', 'u'][pos]
                    if d[key] == digit:
                        recovered.append((digit, i, '刚回补', i / 9.0))
                        break
        results[label] = sorted(recovered, key=lambda x: -x[1])[:3]
    return results


def get_miss_trend(ctx, data, pos_idx):
    """比较近10期和近30期的遗漏趋势，找出正在转热/转冷的数字"""
    # 近10期miss
    miss_10 = {}
    for digit in range(10):
        for i, d in enumerate(data[:10]):
            key = ['h', 't', 'u'][pos_idx]
            if d[key] == digit:
                miss_10[digit] = i
                break
        else:
            miss_10[digit] = 10
    
    # 当前miss（全部数据）
    current = ctx['miss'][pos_idx]
    
    # 找出：当前miss高但近10期出现过的（正在转热）
    turning_hot = []
    for digit in range(10):
        cur = current.get(digit, 0)
        m10 = miss_10.get(digit, 10)
        if cur >= 15 and m10 <= 5:
            turning_hot.append((digit, cur, m10))
    
    return turning_hot


def get_sum_tail_miss_signals(ctx, threshold=10):
    """获取和尾遗漏信号列表"""
    signals = []
    for tail in range(10):
        m = ctx['sum_tail_miss'].get(tail, 0)
        if m >= threshold:
            ratio = m / 9.0
            if m >= 18:
                strength = '强信号'
            elif m >= 14:
                strength = '中等信号'
            else:
                strength = '弱信号'
            signals.append((tail, m, strength, ratio))
    signals.sort(key=lambda x: -x[1])
    return signals


def get_moderate_sum_tail(ctx, low=8, high=20):
    """获取中等遗漏的和尾"""
    signals = []
    for tail in range(10):
        m = ctx['sum_tail_miss'].get(tail, 0)
        if low <= m <= high:
            signals.append((tail, m, '回补窗口', m / 9.0))
    signals.sort(key=lambda x: -x[1])
    return signals


def get_span_miss_signals(ctx, threshold=8):
    """获取跨度遗漏信号列表"""
    signals = []
    for sp in range(10):
        m = ctx['span_miss'].get(sp, 0)
        if m >= threshold:
            ratio = m / 9.0
            if m >= 15:
                strength = '强信号'
            elif m >= 10:
                strength = '中等信号'
            else:
                strength = '弱信号'
            signals.append((sp, m, strength, ratio))
    signals.sort(key=lambda x: -x[1])
    return signals


def get_moderate_span(ctx, low=6, high=18):
    """获取中等遗漏的跨度"""
    signals = []
    for sp in range(10):
        m = ctx['span_miss'].get(sp, 0)
        if low <= m <= high:
            signals.append((sp, m, '回补窗口', m / 9.0))
    signals.sort(key=lambda x: -x[1])
    return signals


def get_hot_numbers(ctx, top_n=3):
    """获取各位热号"""
    result = {}
    for pos, label in enumerate(['百位', '十位', '个位']):
        hot = [x[0] for x in ctx['freq20'][pos].most_common(top_n)]
        result[label] = '/'.join(str(x) for x in hot)
    return result


def get_cold_numbers(ctx, top_n=3):
    """获取各位冷号"""
    result = {}
    for pos, label in enumerate(['百位', '十位', '个位']):
        cold = [x[0] for x in ctx['freq20'][pos].most_common()[-top_n:]]
        result[label] = '/'.join(str(x) for x in cold)
    return result


def get_road_pattern(ctx):
    """012路走势分析"""
    parts = []
    for pos, label in enumerate(['百位', '十位', '个位']):
        road_cnt = ctx['road20'][pos]
        total = sum(road_cnt.values()) or 1
        dominant = road_cnt.most_common(1)[0]
        pct = dominant[1] / total * 100
        parts.append(f"{label}{dominant[0]}路占{pct:.0f}%")
    return '，'.join(parts)


def get_companion_pairs(data, top_n=3):
    """伴随号分析（近期经常一起出现的数字对）"""
    pairs = Counter()
    for d in data[:30]:
        nums = [d['h'], d['t'], d['u']]
        for i in range(3):
            for j in range(i+1, 3):
                pair = tuple(sorted([nums[i], nums[j]]))
                pairs[pair] += 1
    top_pairs = pairs.most_common(top_n)
    return [f"{a}{b}({c}次)" for (a, b), c in top_pairs]


def get_position_trend(ctx):
    """各位近10期vs近20期频率变化趋势"""
    parts = []
    for pos, label in enumerate(['百位', '十位', '个位']):
        top10 = ctx['freq20'][pos].most_common(2)
        if top10:
            digit, cnt = top10[0]
            parts.append(f"{label}{digit}({cnt}次/20期)")
    return '，'.join(parts)


def format_3d_report(data, name, selector_output, analysis_ctx, gold_silver):
    """
    格式化3D/排三的完整分析报告。
    
    selector_output: SmartNumberSelector.analyze_and_recommend() 的返回值 (selected, ctx)
    analysis_ctx: compute_analysis_context() 的返回值（用于6个bullet）
    gold_silver: (gold_tuple, silver_tuple) 金码银码
    """
    selected, _ = selector_output
    n = len(data)
    
    # 最新一期
    latest = data[0]
    latest_period = latest['period']
    latest_date = latest['date']
    latest_nums = f"{latest['h']}{latest['t']}{latest['u']}"
    
    # 计算待开期号
    try:
        latest_num = int(latest_period)
        next_period = str(latest_num + 1)
    except:
        next_period = '???'
    
    # 起始期号
    start_period = data[-1]['period']
    
    # 金码银码
    gold, silver = gold_silver
    # V3 的 gold/silver 各为单个三位数号码；此处若照 V4 的「两注」写法（A / B、B / A）
    # 会把同一对号码正反各显示一遍，出现「金码 128/629、银码 629/128」的重复输出。
    gold_str = f"{gold[0]}{gold[1]}{gold[2]}"
    silver_str = f"{silver[0]}{silver[1]}{silver[2]}"
    
    # 使用 analysis_ctx
    ctx = analysis_ctx
    
    # Build report
    lines = []
    lines.append(f"### 📊 {name}")
    lines.append(f"**最新开奖：{latest_period}期（{latest_date}）→ {latest_nums}**")
    lines.append(f"**待开：{next_period}期**")
    lines.append(f"**🥇 金码：{gold_str}**")
    lines.append(f"**🥈 银码：{silver_str}**")
    
    # 10注推荐表格
    lines.append(f"**📋 10注推荐：**")
    lines.append(f"| 序号 | 号码 | 和值 | 跨度 |")
    lines.append(f"|:---:|:---:|:---:|:---:|")
    
    for i, item in enumerate(selected[:10]):
        b, s, g = item['b'], item['s'], item['g']
        hz = item['hz']
        kd = item['kd']
        lines.append(f"| {i+1} | **{b}{s}{g}** | {hz} | {kd} |")
    
    # 分析逻辑（6个bullet，每次随机选择不同展示模式）
    lines.append(f"**分析逻辑：**")
    
    # ── Bullet 1: 遗漏回补（3种稳定模式，按遗漏强度自动选择）──
    miss_signals = get_miss_signals(ctx, threshold=15)
    moderate_miss = get_moderate_miss_signals(ctx, low=10, high=25)
    recovered = get_recent_recovered(ctx, data, lookback=10)
    
    # 自动选择模式：有极端遗漏→展示top2，有中等遗漏→展示回补窗口，否则→全局排名
    has_extreme = any(len(sigs) > 0 and sigs[0][1] >= 20 for sigs in miss_signals.values())
    has_moderate = any(len(sigs) > 0 for sigs in moderate_miss.values())
    has_recovery = any(len(recs) > 0 for recs in recovered.values())
    
    # 使用 advanced 中的 Z-score 来增强遗漏描述的科学性
    adv = ctx.get('advanced', {})
    
    if has_extreme:
        # 模式A：展示最强遗漏top2，附Z-score
        parts = []
        for label, pos_idx in [('百位', 0), ('十位', 1), ('个位', 2)]:
            sigs = miss_signals[label]
            if len(sigs) >= 1:
                d1, m1, s1, r1 = sigs[0]
                z = adv.get('miss_zscore', [{}]*3)[pos_idx].get(d1, 0)
                z_text = f"Z={z:.1f}" if abs(z) > 0.5 else ""
                parts.append(f"{label}**{d1}**遗漏**{m1}期**（{r1:.1f}倍{z_text and '，'+z_text}）→ {s1}")
            else:
                parts.append(f"{label}暂无强信号")
        lines.append(f"- 遗漏回补：{'；'.join(parts)}")

    elif has_moderate:
        # 模式B：中等遗漏回补窗口
        parts = []
        for label, pos_idx in [('百位', 0), ('十位', 1), ('个位', 2)]:
            sigs = moderate_miss[label]
            if sigs:
                d, m, _, ratio = sigs[0]
                z = adv.get('miss_zscore', [{}]*3)[pos_idx].get(d, 0)
                z_text = f"Z={z:.1f}" if abs(z) > 0.5 else ""
                parts.append(f"{label}**{d}**遗漏**{m}期**（{ratio:.1f}倍{z_text and '，'+z_text}）→ 回补窗口")
            else:
                parts.append(f"{label}暂无信号")
        lines.append(f"- 遗漏回补：{'；'.join(parts)}")

    elif has_recovery:
        # 模式C：近期回补信号
        parts = []
        for label, pos_idx in [('百位', 0), ('十位', 1), ('个位', 2)]:
            recs = recovered[label]
            if recs:
                d, prev_miss, _, _ = recs[0]
                parts.append(f"{label}**{d}**刚回补（此前遗漏{prev_miss}期）")
            else:
                sigs = miss_signals[label]
                if sigs:
                    d, m, strength, ratio = sigs[0]
                    parts.append(f"{label}**{d}**遗漏**{m}期**（{ratio:.1f}倍）→ {strength}")
                else:
                    parts.append(f"{label}暂无信号")
        lines.append(f"- 遗漏回补：{'；'.join(parts)}")

    else:
        # 模式D：全局排序top3
        all_miss = []
        for label in ['百位', '十位', '个位']:
            for digit, m, strength, ratio in miss_signals[label]:
                all_miss.append((digit, m, strength, ratio, label))
        all_miss.sort(key=lambda x: -x[1])
        if all_miss[:3]:
            parts = [f"{label}**{d}**遗漏**{m}期**（{r:.1f}倍）→ {s}"
                     for d, m, s, r, label in all_miss[:3]]
            lines.append(f"- 遗漏回补：{'；'.join(parts)}")
        else:
            lines.append(f"- 遗漏回补：暂无强信号（各位置遗漏均在正常范围内）")
    
    # ── Bullet 2: 和尾遗漏（稳定模式：有强信号展示强信号，否则展示top2）──
    sum_tail_sigs = get_sum_tail_miss_signals(ctx, threshold=10)
    moderate_tail = get_moderate_sum_tail(ctx, low=8, high=20)
    
    if sum_tail_sigs and sum_tail_sigs[0][1] >= 14:
        # 有强信号：展示最强的
        tail, miss, strength, ratio = sum_tail_sigs[0]
        lines.append(f"- 和尾遗漏：和尾**{tail}**遗漏**{miss}期**（均值9期，{ratio:.1f}倍）→ {strength}，回补方向")
    elif len(sum_tail_sigs) >= 2:
        # 有多个信号：展示top2
        parts = []
        for tail, miss, strength, ratio in sum_tail_sigs[:2]:
            parts.append(f"和尾**{tail}**遗漏**{miss}期**（{ratio:.1f}倍）→ {strength}")
        lines.append(f"- 和尾遗漏：{'；'.join(parts)}")
    elif sum_tail_sigs:
        # 只有弱信号
        tail, miss, strength, ratio = sum_tail_sigs[0]
        lines.append(f"- 和尾遗漏：和尾**{tail}**遗漏**{miss}期**（均值9期，{ratio:.1f}倍）→ {strength}，回补方向")
    else:
        # 没有信号，展示遗漏最大的
        max_tail_miss = max(ctx['sum_tail_miss'].items(), key=lambda x: x[1])
        lines.append(f"- 和尾遗漏：和尾{max_tail_miss[0]}遗漏{max_tail_miss[1]}期，暂无强回补信号")
    
    # ── Bullet 3: 跨度遗漏（稳定模式：有强信号展示强信号，否则展示top2）──
    span_sigs = get_span_miss_signals(ctx, threshold=8)
    moderate_sp = get_moderate_span(ctx, low=6, high=18)
    
    if span_sigs and span_sigs[0][1] >= 12:
        # 有强信号：展示最强的
        sp, miss, strength, ratio = span_sigs[0]
        lines.append(f"- 跨度遗漏：跨度**{sp}**遗漏**{miss}期**（均值9期，{ratio:.1f}倍）→ {strength}，回补方向")
    elif len(span_sigs) >= 2:
        # 有多个信号：展示top2
        parts = []
        for sp, miss, strength, ratio in span_sigs[:2]:
            parts.append(f"跨度**{sp}**遗漏**{miss}期**（{ratio:.1f}倍）→ {strength}")
        lines.append(f"- 跨度遗漏：{'；'.join(parts)}")
    elif span_sigs:
        # 只有弱信号
        sp, miss, strength, ratio = span_sigs[0]
        lines.append(f"- 跨度遗漏：跨度**{sp}**遗漏**{miss}期**（均值9期，{ratio:.1f}倍）→ {strength}，回补方向")
    else:
        # 没有信号，展示遗漏最大的
        max_span_miss = max(ctx['span_miss'].items(), key=lambda x: x[1])
        lines.append(f"- 跨度遗漏：跨度{max_span_miss[0]}遗漏{max_span_miss[1]}期，暂无强回补信号")
    
    # ── Bullet 4: 热号+冷号+012路+伴随号（稳定模式：综合展示）──
    hot = get_hot_numbers(ctx, top_n=3)
    cold = get_cold_numbers(ctx, top_n=3)
    road = get_road_pattern(ctx)
    pairs = get_companion_pairs(data, top_n=3)
    trend = get_position_trend(ctx)
    
    # 主行：热号（始终展示）
    lines.append(f"- 热号支撑：百位{hot['百位']}，十位{hot['十位']}，个位{hot['个位']}")
    
    # 冷号行：如果存在明显冷号
    cold_parts = []
    for pos, label in enumerate(['百位', '十位', '个位']):
        # 取 freq20 末位且 miss >= 12
        cold_list = [x[0] for x in ctx['freq20'][pos].most_common()[-3:] if ctx['miss'][pos].get(x[0], 0) >= 12]
        if cold_list:
            cold_parts.append(f"{label}{'/'.join(str(x) for x in cold_list)}")
    if cold_parts:
        lines.append(f"- 冷号防守：{'，'.join(cold_parts)}")
    
    # 伴随号行（如果热门对子出现>=2次）
    if pairs and int(pairs[0].split('(')[1].split('次')[0]) >= 2:
        lines.append(f"- 伴随号：{'、'.join(pairs[:2])}")
    
    # 附加行：012路走势（如果有明显偏向）
    road_parts = []
    for pos, label in enumerate(['百位', '十位', '个位']):
        road_cnt = ctx['road20'][pos]
        total = sum(road_cnt.values()) or 1
        dominant = road_cnt.most_common(1)[0]
        pct = dominant[1] / total * 100
        if pct >= 45:  # 只有明显偏向才展示
            road_parts.append(f"{label}{dominant[0]}路占{pct:.0f}%")
    if road_parts:
        lines.append(f"- 012路走势：{'，'.join(road_parts)}")
    
    # ── Bullet 5: 形态统计（稳定模式：始终展示计数+比例）──
    zuliu = ctx['shape_cnt'].get('组六', 0)
    zusan = ctx['shape_cnt'].get('组三', 0)
    baozi = ctx['shape_cnt'].get('豹子', 0)
    total = zuliu + zusan + baozi or 1
    
    lines.append(f"- 形态统计：组六{zuliu}次（{zuliu/total*100:.0f}%）、组三{zusan}次（{zusan/total*100:.0f}%）、豹子{baozi}次")
    
    # ── Bullet 6: 奇偶/大小比（稳定模式：始终展示top2）──
    top_oe = ctx['odd_even_cnt'].most_common(2) if ctx['odd_even_cnt'] else [('1:2', 0)]
    top_bs = ctx['big_small_cnt'].most_common(2) if ctx['big_small_cnt'] else [('1:2', 0)]
    
    oe_str = '、'.join(f"{k}({v}次)" for k, v in top_oe)
    bs_str = '、'.join(f"{k}({v}次)" for k, v in top_bs)
    lines.append(f"- 奇偶分布：{oe_str}；大小分布：{bs_str}")
    
    return '\n'.join(lines)


def format_dlt_report(data, selected, analyzer):
    """
    格式化大乐透的完整分析报告。
    使用 dlt_analysis.py 的 DLTAnalyzer 和 format_report。
    去掉原脚本自带的footer（由主脚本统一添加）。
    """
    from dlt_analysis import format_report
    raw = format_report(analyzer, selected)
    # 去掉末尾的 disclaimer 行（以 "> ⚠️" 或 "> 数据来源" 开头的行）
    lines = raw.split('\n')
    filtered = []
    for line in lines:
        if line.startswith('> ⚠️') or line.startswith('> 数据来源') or line.startswith('> 分析维度'):
            continue
        filtered.append(line)
    return '\n'.join(filtered).rstrip()


def format_3d_report_v4(data, name, rec, analysis_ctx):
    """
    格式化 3D/排三 V4 / ensemble 报告。
    rec 格式: {'gold': [...], 'silver': [...], 'top10': [...], 'style_used': str, 'backtest_hint': str}
    top10 元素建议: {'num': (b,s,g), 'sum': int, 'span': int, 'score': float, 'details': dict, 'group_type': str}
    """
    latest = data[0]
    latest_period = latest['period']
    latest_date = latest.get('date', '')
    latest_nums = f"{latest['h']}{latest['t']}{latest['u']}"
    try:
        next_period = str(int(latest_period) + 1)
    except Exception:
        next_period = '???'

    gold = rec.get('gold', [(0, 0, 0), (0, 0, 0)])
    silver = rec.get('silver', [(0, 0, 0), (0, 0, 0)])
    gold_str = ' / '.join(f"{a}{b}{c}" for a, b, c in gold[:2])
    silver_str = ' / '.join(f"{a}{b}{c}" for a, b, c in silver[:2])

    lines = []
    lines.append(f"### 📊 {name}")
    lines.append(f"**最新开奖：{latest_period}期（{latest_date}）→ {latest_nums}**")
    lines.append(f"**待开：{next_period}期**")
    lines.append(f"**🥇 金码：{gold_str}**")
    lines.append(f"**🥈 银码：{silver_str}**")
    lines.append(f"**📋 10注推荐：**")
    lines.append(f"| 序号 | 号码 | 和值 | 跨度 |")
    lines.append(f"|:---:|:---:|:---:|:---:|")

    top10 = rec.get('top10', [])
    for i, item in enumerate(top10[:10]):
        if isinstance(item, dict):
            nums = item.get('num')
            if isinstance(nums, tuple) and len(nums) == 3:
                b, s, g = nums
            else:
                b = item.get('b', 0)
                s = item.get('s', 0)
                g = item.get('g', 0)
            hz = item.get('sum', b + s + g)
            kd = item.get('span', max(b, s, g) - min(b, s, g))
        else:
            b, s, g = item[:3]
            hz = b + s + g
            kd = max(b, s, g) - min(b, s, g)
        lines.append(f"| {i+1} | **{b}{s}{g}** | {hz} | {kd} |")

    # V4 统计说明 bullet
    style_used = rec.get('style_used', 'balanced')
    bt_hint = rec.get('backtest_hint', '')
    lines.append(f"**分析逻辑（V4）：**")
    lines.append(f"- 模型说明：本注基于 Markov 转移 + 联合分布 + 几何遗漏的自适应评分，当前风格 **{style_used}**")
    if bt_hint:
        lines.append(f"- 回测参考：{bt_hint}")
    lines.append(f"- 理论命中：直选 1/1000，组六约 1/167，组三约 1/333；分析仅提升选号质量，不提高中奖概率")

    # 从 data 直接计算基础统计 bullet（兼容 V3/V4 context）
    n = len(data)
    # 热号：近 20 期各位置出现最多的数字
    hot_parts = []
    pos_keys = [('h', '百位'), ('t', '十位'), ('u', '个位')]
    for key, label in pos_keys:
        cnt = Counter(d[key] for d in data[:20])
        top = cnt.most_common(1)[0][0]
        hot_parts.append(f"{label}{top}")
    lines.append(f"- 热号支撑：{'，'.join(hot_parts)}")

    # 形态统计
    shape_cnt = Counter()
    for d in data:
        nums = [d['h'], d['t'], d['u']]
        if len(set(nums)) == 1:
            shape_cnt['豹子'] += 1
        elif len(set(nums)) == 2:
            shape_cnt['组三'] += 1
        else:
            shape_cnt['组六'] += 1
    total = sum(shape_cnt.values()) or 1
    lines.append(f"- 形态统计：组六{shape_cnt['组六']}次（{shape_cnt['组六']/total*100:.0f}%）、组三{shape_cnt['组三']}次（{shape_cnt['组三']/total*100:.0f}%）、豹子{shape_cnt['豹子']}次")

    # 奇偶/大小
    oe_cnt = Counter()
    bs_cnt = Counter()
    for d in data:
        nums = [d['h'], d['t'], d['u']]
        odd = sum(1 for x in nums if x % 2 == 1)
        big = sum(1 for x in nums if x >= 5)
        oe_cnt[f"{odd}:{3-odd}"] += 1
        bs_cnt[f"{big}:{3-big}"] += 1
    oe_str = '、'.join(f"{k}({v}次)" for k, v in oe_cnt.most_common(2))
    bs_str = '、'.join(f"{k}({v}次)" for k, v in bs_cnt.most_common(2))
    lines.append(f"- 奇偶分布：{oe_str}；大小分布：{bs_str}")

    return '\n'.join(lines)


def _dlt_calc_ac(combo):
    """计算前区 AC 值（算术复杂度）。
    AC = |{ |aᵢ-aⱼ| : i<j }| - (n-1)；AC≥4 为合理，5-8 最常见。"""
    combo = list(combo)
    diffs = set()
    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            diffs.add(abs(combo[i] - combo[j]))
    return len(diffs) - (len(combo) - 1)


def _dlt_zone_ratio(front):
    """三区分布比：1-12 / 13-23 / 24-35。"""
    z = [0, 0, 0]
    for x in front:
        if x <= 12:
            z[0] += 1
        elif x <= 23:
            z[1] += 1
        else:
            z[2] += 1
    return f"{z[0]}:{z[1]}:{z[2]}"


def _dlt_bet_metrics(front):
    """补齐大乐透单注的展示指标（AC/跨度/奇偶/三区）。

    用于 V4 与 ensemble 路径：上游未附带这些字段时按号码现算，
    避免报告表格出现全 0。
    """
    front = list(front)
    if not front:
        return {'ac': 0, 'span': 0, 'odd_even': '0:0', 'three_zone': '0:0:0'}
    odd = sum(1 for x in front if x % 2 == 1)
    return {
        'ac': _dlt_calc_ac(front),
        'span': max(front) - min(front),
        'odd_even': f"{odd}:{len(front) - odd}",
        'three_zone': _dlt_zone_ratio(front),
    }


def format_dlt_report_v4(data, rec, analyzer):
    """
    格式化大乐透 V4 / ensemble 报告。
    rec 格式: {'gold': {...}, 'silver': {...}, 'top5': [...], 'style_used': str, 'backtest_hint': str}
    top5 元素: {'front': [...], 'back': [...], 'front_sum': int, 'back_sum': int, 'ac': int,
                'span': int, 'odd_even': str, 'three_zone': str, 'score': float, 'details': dict}
    """
    latest = data[0]
    latest_period = latest['period']
    latest_date = latest.get('date', '')
    front_str = ' '.join(f"{x:02d}" for x in latest['front'])
    back_str = ' '.join(f"{x:02d}" for x in latest['back'])
    try:
        next_period = str(int(latest_period) + 1)
    except Exception:
        next_period = '???'

    gold = rec.get('gold', {'front': [0]*5, 'back': [0]*2})
    silver = rec.get('silver', {'front': [0]*5, 'back': [0]*2})
    gold_front = ' '.join(f"{x:02d}" for x in gold.get('front', [0]*5))
    gold_back = ' '.join(f"{x:02d}" for x in gold.get('back', [0]*2))
    silver_front = ' '.join(f"{x:02d}" for x in silver.get('front', [0]*5))
    silver_back = ' '.join(f"{x:02d}" for x in silver.get('back', [0]*2))

    lines = []
    lines.append(f"### 📊 大乐透")
    lines.append(f"**最新开奖：{latest_period}期（{latest_date}）→ 前区 {front_str} 后区 {back_str}**")
    lines.append(f"**待开：{next_period}期**")
    lines.append(f"**🥇 金码：前区 {gold_front} / 后区 {gold_back}**")
    lines.append(f"**🥈 银码：前区 {silver_front} / 后区 {silver_back}**")
    lines.append(f"**📋 5注精选推荐：**")
    lines.append(f"| 序号 | 前区号码 | 后区 | 和值 | AC | 跨度 | 奇偶 | 三区 | 分数 |")
    lines.append(f"|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    top5 = rec.get('top5', [])
    for i, item in enumerate(top5[:5]):
        front = item.get('front', [0]*5)
        back = item.get('back', [0]*2)
        fstr = ' '.join(f"{x:02d}" for x in front)
        bstr = ' '.join(f"{x:02d}" for x in back)
        # 上游（V4 / ensemble）可能未附带展示指标，缺失时按号码现算兜底
        m = _dlt_bet_metrics(front)
        hz = item.get('front_sum') or sum(front)
        ac = item.get('ac') or m['ac']
        span = item.get('span') or m['span']
        oe = item.get('odd_even') or m['odd_even']
        tz = item.get('three_zone') or m['three_zone']
        try:
            score = float(item.get('score') or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        lines.append(f"| {i+1} | **{fstr}** | {bstr} | {hz} | {ac} | {span} | {oe} | {tz} | {score:.1f} |")

    style_used = rec.get('style_used', 'balanced')
    bt_hint = rec.get('backtest_hint', '')
    lines.append(f"**分析逻辑（V4）：**")
    lines.append(f"- 模型说明：基于 Copula/共现网络 + 遗传算法 + 后区 Markov 的自适应评分，风格 **{style_used}**")
    if bt_hint:
        lines.append(f"- 回测参考：{bt_hint}")
    lines.append(f"- 理论命中：一等奖约 1/21,425,712；分析仅优化组合分布，不改变中奖概率")
    return '\n'.join(lines)


def _extract_bsg(item):
    """从 V4 / ensemble 的候选元素中统一取出 (百, 十, 个)。"""
    if isinstance(item, dict):
        nums = item.get('num')
        if isinstance(nums, (tuple, list)) and len(nums) == 3:
            return int(nums[0]), int(nums[1]), int(nums[2])
        return int(item.get('b', 0)), int(item.get('s', 0)), int(item.get('g', 0))
    return int(item[0]), int(item[1]), int(item[2])


def apply_quality_floor(data, candidates, style='balanced'):
    """
    V4 / ensemble 路径的质量底线后处理。

    所有 3D/排三清单必须经过 smart_selector.enforce_gold_silver() 出口，
    否则会出现「0-9 没覆盖全」「某个数字占了 7 注」这类隐性集中。
    流程：质量底线过滤（跨度3-8、和值6-23）→ 组合去重 → 金银码占位
          → 0-9 全覆盖 → 单数字出现注数 ≤ 5。
    """
    smart = SmartNumberSelector()
    ctx = smart.compute_context(data)

    items = []
    for cand in candidates[:10]:
        b, s, g = _extract_bsg(cand)
        hz, kd = b + s + g, max(b, s, g) - min(b, s, g)
        if not (3 <= kd <= 8 and 6 <= hz <= 23):
            continue  # 不合质量底线的直接不入池，后续由候选池补足
        sc, details = smart.score_candidate_v3(b, s, g, ctx, style)
        uniq = len({b, s, g})
        items.append({
            'score': sc, 'b': b, 's': s, 'g': g, 'hz': hz, 'kd': kd,
            'shape': 'baozi' if uniq == 1 else ('zusan' if uniq == 2 else 'zuliu'),
            'odd': sum(1 for x in (b, s, g) if x % 2 == 1),
            'big': sum(1 for x in (b, s, g) if x >= 5),
            'details': details,
        })

    gold = (items[0]['b'], items[0]['s'], items[0]['g'])
    # 银码：优先取与金码数字集合交集 ≤ 1 的最高分注。
    # 否则 Borda 融合的高分注天然雷同（如金码 478 / 银码 078 共享 78），
    # 清单看似有金银两码，实际覆盖度只有一组数字。
    gold_set = set(gold)
    silver_cand = next((it for it in items[1:]
                        if len(set((it['b'], it['s'], it['g'])) & gold_set) <= 1), None)
    silver = (silver_cand['b'], silver_cand['s'], silver_cand['g']) if silver_cand else gold

    fixed = smart.enforce_gold_silver(items, ctx, gold, silver, style=style)
    return {'items': fixed, 'gold': gold, 'silver': silver}


def run_single_game(game_code, count=200, version='v3'):
    """
    运行单个彩种的完整分析流程。
    version: 'v3' | 'v4' | 'ensemble'
    返回格式化的报告文本。
    """
    if game_code == '3d':
        name = '福彩3D'
        data = fetch_3d(count)
        print(f"[{name}] 获取到 {len(data)} 期数据", file=sys.stderr)

        if version == 'v4' and SmartNumberSelectorV4 is not None:
            smart4 = SmartNumberSelectorV4(style='adaptive')
            rec = smart4.recommend_v4(data, n=10)
            analysis_ctx = smart4.compute_context(data)
            # 质量底线后处理：V4 输出同样必须走 enforce_gold_silver 出口
            qf = apply_quality_floor(data, rec.get('top10', [])[:10])
            fixed = qf['items']
            if fixed:
                rec['top10'] = [{'b': it['b'], 's': it['s'], 'g': it['g'],
                                 'sum': it['hz'], 'span': it['kd']} for it in fixed]
                rec['gold'] = [qf['gold']]
                rec['silver'] = [qf['silver']]
            report = format_3d_report_v4(data, name, rec, analysis_ctx)
            return report, data
        elif version == 'ensemble' and EnsembleRecommender3D is not None:
            raw = EnsembleRecommender3D(data, style='adaptive').recommend(n=10)
            bets = raw.get('bets', [])
            # 质量底线后处理：ensemble 输出必须走 enforce_gold_silver 出口，
            # 否则 Borda 融合结果会出现 0-9 未覆盖 / 单数字占用过多注数的问题
            qf = apply_quality_floor(data, bets[:10])
            fixed = qf['items']
            rec = {
                'gold': [qf['gold']],
                'silver': [qf['silver']],
                'top10': [{'b': it['b'], 's': it['s'], 'g': it['g'],
                           'sum': it['hz'], 'span': it['kd']} for it in fixed],
                'style_used': raw.get('style', 'balanced'),
                'backtest_hint': 'V3+V4 Ensemble 集成推荐，来源：' + '+'.join(raw.get('sources', ['v3']))
            }
            report = format_3d_report_v4(data, name, rec, None)
            return report, data
        else:
            # V3 默认路径
            smart = SmartNumberSelector()
            analysis_ctx = compute_analysis_context(data, smart)
            selected, sel_ctx = smart.analyze_and_recommend(data, style='balanced')
            gold_silver = smart.generate_gold_silver(data, sel_ctx, style='balanced')
            # （V3.1）保证金码/银码落在10注清单内：组合级去重 + 0-9 数字覆盖校正
            selected = smart.enforce_gold_silver(selected, sel_ctx, gold_silver[0],
                                                 gold_silver[1], style='balanced')
            report = format_3d_report(data, name, (selected, sel_ctx), analysis_ctx, gold_silver)
            return report, data
        
    elif game_code == '35':
        name = '排列三'
        data = fetch_pl3('35', count)
        print(f"[{name}] 获取到 {len(data)} 期数据", file=sys.stderr)

        if version == 'v4' and SmartNumberSelectorV4 is not None:
            smart4 = SmartNumberSelectorV4(style='adaptive')
            rec = smart4.recommend_v4(data, n=10)
            analysis_ctx = smart4.compute_context(data)
            # 质量底线后处理：V4 输出同样必须走 enforce_gold_silver 出口
            qf = apply_quality_floor(data, rec.get('top10', [])[:10])
            fixed = qf['items']
            if fixed:
                rec['top10'] = [{'b': it['b'], 's': it['s'], 'g': it['g'],
                                 'sum': it['hz'], 'span': it['kd']} for it in fixed]
                rec['gold'] = [qf['gold']]
                rec['silver'] = [qf['silver']]
            report = format_3d_report_v4(data, name, rec, analysis_ctx)
            return report, data
        elif version == 'ensemble' and EnsembleRecommender3D is not None:
            raw = EnsembleRecommender3D(data, style='adaptive').recommend(n=10)
            bets = raw.get('bets', [])
            # 质量底线后处理：ensemble 输出必须走 enforce_gold_silver 出口，
            # 否则 Borda 融合结果会出现 0-9 未覆盖 / 单数字占用过多注数的问题
            qf = apply_quality_floor(data, bets[:10])
            fixed = qf['items']
            rec = {
                'gold': [qf['gold']],
                'silver': [qf['silver']],
                'top10': [{'b': it['b'], 's': it['s'], 'g': it['g'],
                           'sum': it['hz'], 'span': it['kd']} for it in fixed],
                'style_used': raw.get('style', 'balanced'),
                'backtest_hint': 'V3+V4 Ensemble 集成推荐，来源：' + '+'.join(raw.get('sources', ['v3']))
            }
            report = format_3d_report_v4(data, name, rec, None)
            return report, data
        else:
            smart = SmartNumberSelector()
            analysis_ctx = compute_analysis_context(data, smart)
            selected, sel_ctx = smart.analyze_and_recommend(data, style='balanced')
            gold_silver = smart.generate_gold_silver(data, sel_ctx, style='balanced')
            # （V3.1）保证金码/银码落在10注清单内：组合级去重 + 0-9 数字覆盖校正
            selected = smart.enforce_gold_silver(selected, sel_ctx, gold_silver[0],
                                                 gold_silver[1], style='balanced')
            report = format_3d_report(data, name, (selected, sel_ctx), analysis_ctx, gold_silver)
            return report, data
        
    elif game_code == '85':
        name = '大乐透'
        data = fetch_dlt(count)
        print(f"[{name}] 获取到 {len(data)} 期数据", file=sys.stderr)

        if version == 'v4' and DLTAnalyzerV4 is not None:
            analyzer = DLTAnalyzerV4(data, style='adaptive')
            rec = analyzer.recommend_v4(n=5)
            report = format_dlt_report_v4(data, rec, analyzer)
            return report, data
        elif version == 'ensemble' and EnsembleRecommenderDLT is not None:
            raw = EnsembleRecommenderDLT(data, style='adaptive').recommend(n=5)
            bets = raw.get('bets', [])
            rec = {
                'gold': {'front': bets[0]['front'], 'back': bets[0]['back']} if bets else {'front': [0]*5, 'back': [0]*2},
                'silver': {'front': bets[1]['front'], 'back': bets[1]['back']} if len(bets) > 1 else {'front': [0]*5, 'back': [0]*2},
                'top5': [{'front': b['front'], 'back': b['back'],
                          'front_sum': sum(b['front']), 'back_sum': sum(b['back']),
                          'score': b.get('score', 0.0)}
                         for b in bets[:5]],
                'style_used': raw.get('style', 'balanced'),
                'backtest_hint': 'V3+V4 Ensemble 集成推荐，来源：' + '+'.join(raw.get('sources', ['v3']))
            }
            report = format_dlt_report_v4(data, rec, None)
            return report, data
        else:
            from dlt_analysis import DLTAnalyzer
            analyzer = DLTAnalyzer(data, style='balanced')
            selected = analyzer.recommend(5)
            report = format_dlt_report(data, selected, analyzer)
            return report, data
        
    else:
        raise ValueError(f"不支持的彩种代码: {game_code}")


def main():
    parser = argparse.ArgumentParser(description='一体化彩票分析脚本')
    parser.add_argument('--games', type=str, required=True,
                        help='彩种列表，逗号分隔: 3d,35,85')
    parser.add_argument('--count', type=int, default=200,
                        help='获取数据期数 (默认200)')
    parser.add_argument('--version', type=str, default='v3',
                        choices=['v3', 'v4', 'ensemble'],
                        help='模型版本：v3(默认) | v4 | ensemble')
    parser.add_argument('--seed', type=int, default=None,
                        help='固定随机种子，使 v4/ensemble 结果可复现（ensemble 含 Bootstrap 采样，默认每次不同）')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    
    # 统一彩种代码：支持 3d/3D/35/85，也支持用户说的 pl3/排三/P3 等别名
    aliases = {
        '3d': '3d', '3D': '3d', '福彩3d': '3d', '福彩3D': '3d',
        '35': '35', 'pl3': '35', 'PL3': '35', '排列三': '35', '排三': '35',
        '85': '85', 'dlt': '85', 'DLT': '85', '大乐透': '85',
    }
    # 兼容 shell/CLI 把逗号列表拆成多个参数的情况
    raw_input = str(args.games)
    # 兼容某些执行环境把 '3d' 解析成 '3' 导致不支持的彩种代码错误
    cleaned_parts = [p.strip() for p in raw_input.split(',') if p.strip()]
    restored = []
    for p in cleaned_parts:
        if p == '3':
            restored.append('3d')
        else:
            restored.append(p)
    raw_codes = restored
    game_codes = []
    seen = set()
    for c in raw_codes:
        normalized = aliases.get(c, c)
        if normalized not in seen:
            seen.add(normalized)
            game_codes.append(normalized)
    
    # 按固定顺序排列: 3d -> 35 -> 85
    order = {'3d': 0, '35': 1, '85': 2}
    game_codes.sort(key=lambda x: order.get(x, 99))
    
    reports = []
    all_data = {}
    name_map = {'3d': '福彩3D', '35': '排列三', '85': '大乐透'}
    
    for code in game_codes:
        try:
            report, data = run_single_game(code, args.count, version=args.version)
            reports.append(report)
            all_data[code] = data
        except Exception as e:
            # 静默失败：不打印堆栈到 stderr，避免污染最终输出
            print(f"[WARN] {name_map.get(code, code)} 跳过: {e}", file=sys.stderr)
            # 失败的彩种不加入 reports，避免报告里出现错误条
    
    # 过滤出成功获取到数据的彩种
    successful_codes = [code for code in game_codes if code in all_data]
    
    # 构建统一header（只包含成功获取到数据的彩种）
    header_parts = []
    for code in successful_codes:
        d = all_data[code]
        name = name_map.get(code, code)
        header_parts.append(f"{name} {d[-1]['period']}–{d[0]['period']}期")
    
    # 合并输出（所有彩种一条消息）
    now = datetime.now()
    actual_count = min(len(all_data[code]) for code in successful_codes) if successful_codes else args.count
    # Dynamic subtitle: 10注 for 3D/pl3, 5注 for dlt
    bet_label = "10注精选"
    if successful_codes == ['85']:
        bet_label = "5注精选"
    elif '85' in successful_codes and len(successful_codes) > 1:
        bet_label = "10注/5注精选"
    version_label = {'v3': 'V3.0', 'v4': 'V4.0', 'ensemble': 'V3+V4 Ensemble'}[args.version]
    output = f"分析完成，基于{actual_count}期数据，{version_label} 多策略评分推荐如下：\n"
    output += "---\n"
    output += f"## 🎰 金码银码推荐 & {bet_label}\n"
    output += f"**{now.strftime('%Y-%m-%d %H:%M')} | {' | '.join(header_parts)}**\n"
    output += "---\n"
    output += "\n---\n".join(reports)
    
    # 如果全部失败，给出友好提示，不再生成空推荐
    if not successful_codes:
        output += "\n---\n"
        output += "> ⚠️ 所有彩种数据获取均失败，请检查网络连接后重试。\n"
        print(output)
        return
    
    # 添加统一footer（成功彩种才加）
    output += "\n---\n"
    output += "> ⚠️ 仅供参考，请理性投注！\n"
    
    # 计算数据来源（只包含成功获取到数据的彩种）
    source_parts = []
    for code in successful_codes:
        d = all_data[code]
        source_parts.append(f"{d[-1]['period']}–{d[0]['period']}(各{len(d)}期)")
    output += f"> 数据来源：官方API {' | '.join(source_parts)}，智能选号系统{version_label}\n"
    output += "> \n"
    if args.version == 'v4':
        output += "> 📊 选号说明：基于 Markov 链 + 联合分布 + 几何遗漏 + 遗传算法/Copula 的 V4 概率统计模型\n"
        output += "> - 3D/排三：Markov 转移、ACF 周期、regime 切换、Bootstrap 置信区间\n"
        output += "> - 大乐透：前区共现网络/PageRank、遗传算法、后区 Markov、同尾/差值模式\n"
    elif args.version == 'ensemble':
        output += "> 📊 选号说明：V3 + V4 Ensemble 集成，兼顾稳定基线与概率统计优化\n"
        output += "> - V3：贝叶斯+指数衰减+复隔中+多策略共识\n"
        output += "> - V4：Markov/Copula/GA/网络分析/回测验证\n"
    else:
        output += "> 📊 选号说明：基于贝叶斯+指数衰减+复隔中+多策略共识的智能选号\n"
    # 统计口径按彩种区分：大乐透（35选5+12选2）不适用 3D/排三 的组六、和值7-20 口径
    has_small = any(c in successful_codes for c in ('3d', '35'))
    has_dlt = '85' in successful_codes
    if has_small:
        output += "> - 3D/排三口径：和值7-20（占83.2%）、跨度4-7（占56.4%）、组六（占72%）\n"
        output += "> - 3D/排三口径：奇偶1:2或2:1（各37.5%）、大小1:2或2:1（各37.5%）\n"
    if has_dlt:
        output += "> - 大乐透口径：前区三区比以 2:2:1 / 2:1:2 / 1:2:2 为主、AC≥4、和值取历史密集区\n"
        output += "> - 大乐透口径：后区 1-12 选 2，参考共现矩阵与一球→二球条件概率\n"
    output += "> - 命中率与随机选号相同，分析只提高选号质量，不预测未来\n"
    output += f"> ⏱️ 生成时间：{now.strftime('%Y-%m-%d %H:%M:%S')}（脚本实时生成，非缓存）\n"
    
    print(output)


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        # 下游分页（如 PowerShell 的 Select-Object -First N）提前关闭管道，
        # 属正常截断而非脚本错误，静默退出避免 exit code 1。
        try:
            sys.stdout.close()
        except Exception:
            pass
        os._exit(0)
