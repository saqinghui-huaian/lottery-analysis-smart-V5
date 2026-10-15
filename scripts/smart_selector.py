#!/usr/bin/env python3
"""
智能选号系统 V3.2 (2026-10-14 融合版)

核心理念：基于多维统计模型的加权推荐，提供有指向性的号码选择

V3.2 融合说明（GitHub V3.0 统计深度 + 本地 V3.1 数学正确性）：
- 基底：GitHub saqinghui-huaian/lottery-analysis-smart-V3（多窗口频率、
  平滑指数衰减、遗漏 Z-score、和尾/跨度 Z-score、形态转移、位置间条件频率）
- 注入：本地 V3.1 的 Dirichlet-Multinomial 共轭后验、输出质量底线、
  组合级去重、金码/银码入注保证、0-9 数字覆盖校正

V3.0 核心改进：
1. 贝叶斯后验概率：Dirichlet(α=3) 共轭先验 → 后验均值
2. 指数衰减加权：exp(-idx/20) 平滑衰减，归一化到 1-5
3. 复隔中分析：专业分析师常用的核心维度
4. 多策略共识：7种策略独立评分（遗漏/和尾/跨度目标采样由 Z-score 驱动）
5. 蒙特卡洛模拟：随机生成+多维过滤
6. 信息熵分析：衡量号码分布的随机性
7. 遗漏统计特征：历史遗漏均值/标准差 → Z-score，识别真正异常遗漏
"""

import random
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional


class SmartNumberSelector:
    """智能选号系统 V3.2"""
    
    # 理论概率分布（枚举计算，精确值）
    SUM_PROBS = {}  # 和值 -> 概率
    SPAN_PROBS = {}  # 跨度 -> 概率
    SHAPE_PROBS = {'baozi': 0.01, 'zusan': 0.27, 'zuliu': 0.72}
    
    # 0-9的质数集合
    PRIMES = {2, 3, 5, 7}
    
    def __init__(self):
        self._compute_theoretical_distributions()
    
    def _compute_theoretical_distributions(self):
        """计算理论概率分布（精确枚举）"""
        sum_count = Counter()
        span_count = Counter()
        
        for b in range(10):
            for s in range(10):
                for g in range(10):
                    hz = b + s + g
                    span = max(b, s, g) - min(b, s, g)
                    sum_count[hz] += 1
                    span_count[span] += 1
        
        # 归一化为概率
        total = 1000
        for hz, cnt in sum_count.items():
            self.SUM_PROBS[hz] = cnt / total
        for span, cnt in span_count.items():
            self.SPAN_PROBS[span] = cnt / total
    
    def compute_context(self, data: List[Dict]) -> Dict:
        ctx = {}
        n = len(data)

        # ── 基础频率统计 ──
        # 全量频率（用于贝叶斯和信息熵）
        freq_all = [Counter(), Counter(), Counter()]
        for d in data:
            for pos, key in enumerate(['h', 't', 'u']):
                freq_all[pos][d[key]] += 1

        # 多窗口频率：近20期、近50期、近100期
        def window_freq(window):
            f = [Counter(), Counter(), Counter()]
            for d in data[:window]:
                for pos, key in enumerate(['h', 't', 'u']):
                    f[pos][d[key]] += 1
            return f

        ctx['freq_all'] = freq_all
        ctx['freq20'] = window_freq(20)
        ctx['freq50'] = window_freq(50)
        ctx['freq100'] = window_freq(100) if n >= 100 else window_freq(n)

        # 指数衰减加权热度（更平滑的衰减，避免近5期权重过高）
        ctx['heat_weighted'] = [Counter(), Counter(), Counter()]
        for idx, d in enumerate(data[:min(n, 100)]):
            # 使用指数衰减：weight = exp(-idx/20)，归一化到 1-5
            w = max(1.0, math.exp(-idx / 20.0) * 5)
            for pos, key in enumerate(['h', 't', 'u']):
                ctx['heat_weighted'][pos][d[key]] += w

        # ── 贝叶斯后验概率（Dirichlet-Multinomial 共轭后验均值）──
        # 先验: Dirichlet(α)，α=3/位（等效30期先验样本，向均匀分布收缩）
        # 后验均值: P(digit) = (count + α) / (N + 10α)
        # V3.0 的拉普拉斯平滑（α=1）先验过弱，小样本仍对噪声过拟合；
        # α=3 时小样本自动向 0.1 收缩，且退化为均匀分布的方向明确。
        ALPHA = 3.0
        ctx['bayesian'] = [{}, {}, {}]
        for pos in range(3):
            total = sum(freq_all[pos].values())
            denom = total + 10 * ALPHA
            for digit in range(10):
                ctx['bayesian'][pos][digit] = (freq_all[pos].get(digit, 0) + ALPHA) / denom

        # ── 遗漏值（科学定义：当前连续未出现的期数）──
        ctx['miss'] = [{}, {}, {}]
        for pos, key in enumerate(['h', 't', 'u']):
            for digit in range(10):
                found = False
                for i, d in enumerate(data):
                    if d[key] == digit:
                        ctx['miss'][pos][digit] = i
                        found = True
                        break
                if not found:
                    ctx['miss'][pos][digit] = n

        # 遗漏的统计特征：每个数字的历史遗漏均值和标准差（判断当前遗漏是否异常）
        ctx['miss_stats'] = {pos: {} for pos in range(3)}
        for pos, key in enumerate(['h', 't', 'u']):
            for digit in range(10):
                gaps = []
                last_idx = None
                for i, d in enumerate(data):
                    if d[key] == digit:
                        if last_idx is not None:
                            gaps.append(i - last_idx - 1)
                        last_idx = i
                if gaps:
                    ctx['miss_stats'][pos][digit] = {
                        'mean': sum(gaps) / len(gaps),
                        'std': math.sqrt(sum((x - sum(gaps)/len(gaps))**2 for x in gaps) / len(gaps)) if len(gaps) > 1 else 3.0,
                        'max': max(gaps)
                    }
                else:
                    ctx['miss_stats'][pos][digit] = {'mean': 9.0, 'std': 8.0, 'max': n}

        # 遗漏强度 = (当前遗漏 - 历史均值) / 历史标准差（Z-score），识别真正异常
        ctx['miss_zscore'] = [{}, {}, {}]
        for pos in range(3):
            for digit in range(10):
                cur = ctx['miss'][pos].get(digit, 0)
                stats = ctx['miss_stats'][pos][digit]
                ctx['miss_zscore'][pos][digit] = (cur - stats['mean']) / max(stats['std'], 1.0)

        # ── 复隔中分布 ──
        if len(data) >= 2:
            # 复：上期出现过的数字；隔：上上期出现但上期未出现；中：更远未出现
            last_nums = set([data[0]['h'], data[0]['t'], data[0]['u']])
            prev_nums = set([data[1]['h'], data[1]['t'], data[1]['u']])
            ctx['repeat_codes'] = last_nums
            ctx['gap_codes'] = prev_nums - last_nums
            ctx['mid_codes'] = set(range(10)) - last_nums - ctx['gap_codes']
        else:
            ctx['repeat_codes'] = set()
            ctx['gap_codes'] = set()
            ctx['mid_codes'] = set(range(10))
        
        # ── 和值/跨度统计 ──
        def compute_value_stats(values, max_val):
            """计算某个离散值的频率和遗漏"""
            freq = Counter(values)
            miss = {}
            for v in range(max_val + 1):
                for i, d in enumerate(data):
                    if values[i] == v:
                        miss[v] = i
                        break
                else:
                    miss[v] = n
            # 历史遗漏统计
            hist_stats = {}
            for v in range(max_val + 1):
                gaps = []
                last_idx = None
                for i, val in enumerate(values):
                    if val == v:
                        if last_idx is not None:
                            gaps.append(i - last_idx - 1)
                        last_idx = i
                if gaps:
                    hist_stats[v] = {
                        'mean': sum(gaps) / len(gaps),
                        'std': math.sqrt(sum((x - sum(gaps)/len(gaps))**2 for x in gaps) / len(gaps)) if len(gaps) > 1 else 3.0,
                    }
                else:
                    hist_stats[v] = {'mean': max(1, n / max_val), 'std': 3.0}
            return freq, miss, hist_stats

        sums_full = [d['h'] + d['t'] + d['u'] for d in data]
        spans_full = [max(d['h'], d['t'], d['u']) - min(d['h'], d['t'], d['u']) for d in data]

        ctx['sum_freq'], ctx['sum_tail_miss'], ctx['sum_hist_stats'] = compute_value_stats(
            [s % 10 for s in sums_full], 9)
        ctx['span_freq'], ctx['span_miss'], ctx['span_hist_stats'] = compute_value_stats(
            spans_full, 9)

        sums_recent = sums_full[:50]
        spans_recent = spans_full[:50]
        ctx['hz_avg'] = sum(sums_recent) / len(sums_recent) if sums_recent else 13.5
        ctx['kd_avg'] = sum(spans_recent) / len(spans_recent) if spans_recent else 4.5

        # 和值/跨度 Z-score（识别异常偏离）
        ctx['sum_tail_zscore'] = {}
        for tail in range(10):
            cur = ctx['sum_tail_miss'].get(tail, 0)
            stats = ctx['sum_hist_stats'][tail]
            ctx['sum_tail_zscore'][tail] = (cur - stats['mean']) / max(stats['std'], 1.0)

        ctx['span_zscore'] = {}
        for sp in range(10):
            cur = ctx['span_miss'].get(sp, 0)
            stats = ctx['span_hist_stats'][sp]
            ctx['span_zscore'][sp] = (cur - stats['mean']) / max(stats['std'], 1.0)

        # ── 形态统计 ──
        ctx['recent_shapes'] = Counter()
        for d in data[:20]:
            unique = len(set([d['h'], d['t'], d['u']]))
            if unique == 1: ctx['recent_shapes']['baozi'] += 1
            elif unique == 2: ctx['recent_shapes']['zusan'] += 1
            else: ctx['recent_shapes']['zuliu'] += 1

        ctx['shape_cnt'] = Counter()
        for d in data[:50]:
            unique = len(set([d['h'], d['t'], d['u']]))
            if unique == 1: ctx['shape_cnt']['豹子'] += 1
            elif unique == 2: ctx['shape_cnt']['组三'] += 1
            else: ctx['shape_cnt']['组六'] += 1

        # 形态转移概率
        ctx['shape_transition'] = {}
        if len(data) >= 2:
            def shape_of(d):
                u = len(set([d['h'], d['t'], d['u']]))
                return 'baozi' if u == 1 else ('zusan' if u == 2 else 'zuliu')
            trans = Counter()
            for i in range(len(data) - 1):
                cur = shape_of(data[i])
                nxt = shape_of(data[i+1])
                trans[(cur, nxt)] += 1
            for cur in ['baozi', 'zusan', 'zuliu']:
                total = sum(trans[(cur, nxt)] for nxt in ['baozi', 'zusan', 'zuliu'])
                if total > 0:
                    ctx['shape_transition'][cur] = {
                        nxt: trans[(cur, nxt)] / total for nxt in ['baozi', 'zusan', 'zuliu']}
                else:
                    ctx['shape_transition'][cur] = {'baozi': 0.01, 'zusan': 0.27, 'zuliu': 0.72}

        # ── 奇偶/大小分布 ──
        ctx['odd_even_cnt'] = Counter()
        ctx['big_small_cnt'] = Counter()
        for d in data[:30]:  # 增加到30期更稳定
            odd = sum(1 for x in [d['h'], d['t'], d['u']] if x % 2 == 1)
            ctx['odd_even_cnt'][f'{odd}:{3-odd}'] += 1
            big = sum(1 for x in [d['h'], d['t'], d['u']] if x >= 5)
            ctx['big_small_cnt'][f'{big}:{3-big}'] += 1

        # ── 012路频率 ──
        ctx['road20'] = [Counter(), Counter(), Counter()]
        for d in data[:30]:
            for pos, key in enumerate(['h', 't', 'u']):
                ctx['road20'][pos][d[key] % 3] += 1
        # 012路当前遗漏：每个位置各路数连续未出现期数
        ctx['road_miss'] = [{}, {}, {}]
        for pos, key in enumerate(['h', 't', 'u']):
            for road in range(3):
                miss = 0
                for i, d in enumerate(data):
                    if d[key] % 3 == road:
                        miss = i
                        break
                else:
                    miss = len(data)
                ctx['road_miss'][pos][road] = miss

        # ── 伴随号（带条件概率）──
        ctx['companion_pairs'] = Counter()
        for d in data[:50]:
            nums = [d['h'], d['t'], d['u']]
            for i in range(3):
                for j in range(i+1, 3):
                    pair = tuple(sorted([nums[i], nums[j]]))
                    ctx['companion_pairs'][pair] += 1

        # 位置间关联：某位出现数字A后，另一位出现数字B的条件频率
        ctx['pair_cond'] = {}
        for pos_a in range(3):
            for pos_b in range(3):
                if pos_a == pos_b: continue
                key_a = ['h', 't', 'u'][pos_a]
                key_b = ['h', 't', 'u'][pos_b]
                cond = {a: Counter() for a in range(10)}
                for d in data:
                    cond[d[key_a]][d[key_b]] += 1
                ctx['pair_cond'][(pos_a, pos_b)] = cond

        # ── 信息熵 ──
        ctx['entropy'] = [0.0, 0.0, 0.0]
        for pos in range(3):
            total = sum(freq_all[pos].values()) or 1
            probs = [freq_all[pos].get(d, 0) / total for d in range(10)]
            ctx['entropy'][pos] = -sum(p * math.log2(p) for p in probs if p > 0)
        ctx['max_entropy'] = math.log2(10)

        # 趋势方向：近10期vs前10期频率变化
        ctx['trend'] = []
        for pos in range(3):
            key = ['h', 't', 'u'][pos]
            recent10 = Counter(d[key] for d in data[:10])
            prev10 = Counter(d[key] for d in data[10:20])
            trending = []
            for digit in range(10):
                r = recent10.get(digit, 0)
                p = prev10.get(digit, 0)
                if r >= 2 and r > p:
                    trending.append((digit, r - p))
            ctx['trend'].append(sorted(trending, key=lambda x: -x[1])[:3])

        ctx['recent_sums'] = sums_full[:20]
        ctx['recent_spans'] = spans_full[:20]

        return ctx
    
    def _position_digit_score(self, pos: int, digit: int, ctx: Dict, style: str) -> float:
        """
        单个位置单个数字的综合得分（频率+趋势+遗漏）。
        返回 [0,1] 区间得分，1 表示最推荐。
        """
        scores = []

        # 1. 多窗口频率得分（近20期权重高，近100期平滑）
        f20 = ctx['freq20'][pos].get(digit, 0) / 20
        f50 = ctx['freq50'][pos].get(digit, 0) / 50
        f100 = ctx['freq100'][pos].get(digit, 0) / max(len(ctx.get('freq100', [[]])[0]) or 100, 100)
        freq_score = 0.5 * f20 + 0.3 * f50 + 0.2 * f100

        # 2. 趋势得分：近10期是否比前10期更热
        trend_bonus = 0
        for td, delta in ctx['trend'][pos]:
            if td == digit:
                trend_bonus = min(delta * 0.15, 0.3)
                break

        # 3. 遗漏得分（核心改进）：使用 Z-score 而非原始遗漏
        # 当前遗漏越大、越超过历史均值，得分越高；但极端遗漏（>3sigma）边际递减
        z = ctx['miss_zscore'][pos].get(digit, 0)
        if z > 0:
            # z 在 [0, 3] 区间映射到 [0, 0.6]
            miss_score = min(z / 3.0, 1.0) * 0.6
        else:
            # 遗漏低于均值，轻微扣分
            miss_score = max(-0.3, z / 3.0 * 0.3)

        # 4. 热度过度惩罚：如果某数字近20期出现次数显著高于期望（≥5次 vs 期望2次），过热反而降权
        if f20 >= 0.25:  # 5/20
            overheat_penalty = -0.15
        elif f20 >= 0.20:
            overheat_penalty = -0.05
        else:
            overheat_penalty = 0

        # 5. 位置间条件概率：利用 pair_cond
        # 这里只返回 digit 本身得分；组合层面再考虑 pair_cond
        total = freq_score + trend_bonus + miss_score + overheat_penalty
        return max(0.0, min(1.0, total))

    def score_candidate_v3(self, b: int, s: int, g: int, ctx: Dict, style: str = 'balanced', data: List[Dict] = None) -> Tuple[float, Dict]:
        details = {}
        score = 0.0

        digits = [b, s, g]

        # ── 1. 位置频率得分（满分20）──
        pos_score = sum(self._position_digit_score(pos, digits[pos], ctx, style) for pos in range(3))
        d1 = pos_score / 3 * 20  # 归一化到20分
        details['位置分'] = round(d1, 1)
        score += d1

        # ── 2. 和值质量（满分15）──
        # 不再简单判断区间，而是结合和尾遗漏与和值历史分布
        hz = b + s + g
        tail = hz % 10
        tail_z = ctx['sum_tail_zscore'].get(tail, 0)
        # 和值在历史中的频率：出现次数越接近理论期望越好，但当前遗漏大的更有价值
        sum_hist_mean = ctx['sum_hist_stats'][tail]['mean']
        sum_cur_miss = ctx['sum_tail_miss'].get(tail, 0)
        # 和值区间基础分
        if 8 <= hz <= 19: base_hz = 12
        elif 6 <= hz <= 21: base_hz = 8
        elif 3 <= hz <= 24: base_hz = 4
        else: base_hz = 0
        # 和尾遗漏加分
        tail_bonus = min(max(tail_z, 0) * 1.5, 3)
        d2 = min(base_hz + tail_bonus, 15)
        details['和值'] = round(d2, 1)
        score += d2

        # ── 3. 跨度质量（满分12）──
        kd = max(b, s, g) - min(b, s, g)
        span_z = ctx['span_zscore'].get(kd, 0)
        if 4 <= kd <= 7: base_kd = 10
        elif 3 <= kd <= 8: base_kd = 7
        elif 2 <= kd <= 9: base_kd = 4
        else: base_kd = 1
        span_bonus = min(max(span_z, 0) * 1.0, 2)
        d3 = min(base_kd + span_bonus, 12)
        details['跨度'] = round(d3, 1)
        score += d3

        # ── 4. 形态得分（满分10）──
        unique = len(set(digits))
        # 根据历史形态转移调整：如果近期组六很热，继续偏好组六
        recent_zuliu_ratio = ctx['recent_shapes'].get('zuliu', 14) / max(sum(ctx['recent_shapes'].values()), 20)
        if unique == 3:
            d4 = 8 + 2 * recent_zuliu_ratio
        elif unique == 2:
            d4 = 5 + 1 * (1 - recent_zuliu_ratio)
        else:
            d4 = 0.5
        details['形态'] = round(d4, 1)
        score += d4

        # ── 5. 奇偶平衡（满分8）──
        odd_count = sum(1 for x in digits if x % 2 == 1)
        # 参考历史分布：通常 1:2 或 2:1 最常见
        if odd_count in [1, 2]: d5 = 8
        else: d5 = 2
        details['奇偶'] = round(d5, 1)
        score += d5

        # ── 6. 大小平衡（满分8）──
        big_count = sum(1 for x in digits if x >= 5)
        if big_count in [1, 2]: d6 = 8
        else: d6 = 2
        details['大小'] = round(d6, 1)
        score += d6

        # ── 7. 012路多样性（满分5）──
        roads = set([x % 3 for x in digits])
        road_score = 0
        if len(roads) == 3:
            road_score = 5
        elif len(roads) == 2:
            road_score = 3
        else:
            road_score = 0
        # 融合路数遗漏：若某路在当前位置连续遗漏>=5期，组合包含该路给予额外奖励
        road_missing_bonus = 0
        for pos, digit in enumerate(digits):
            road = digit % 3
            # 从 ctx 中读取该位置各路数的当前遗漏
            road_miss = ctx.get('road_miss', [{}, {}, {}])[pos].get(road, 0)
            if road_miss >= 5:
                road_missing_bonus += min(road_miss / 5, 2)
        road_score = min(road_score + road_missing_bonus, 8)
        details['012路'] = round(road_score, 1)
        score += road_score

        # ── 8. 复隔中（满分5）──
        nums = set(digits)
        repeat_hit = len(nums & ctx['repeat_codes'])
        gap_hit = len(nums & ctx['gap_codes'])
        # 复隔中本身是短期记忆，权重不宜过高
        d8 = min(repeat_hit * 1.5 + gap_hit * 0.8, 5)
        details['复隔中'] = round(d8, 1)
        score += d8

        # ── 9. 质合平衡（满分4）──
        prime_count = sum(1 for x in digits if x in self.PRIMES)
        if prime_count in [1, 2]: d9 = 4
        else: d9 = 1
        details['质合'] = round(d9, 1)
        score += d9

        # ── 10. 伴随号 / 位置关联（满分6）──
        companion_score = 0
        pair_bonus = 0
        for i in range(3):
            for j in range(i+1, 3):
                pair = tuple(sorted([digits[i], digits[j]]))
                companion_score += ctx['companion_pairs'].get(pair, 0)
                # 位置间条件概率
                cond = ctx['pair_cond'].get((i, j), {})
                if digits[i] in cond:
                    pair_bonus += cond[digits[i]].get(digits[j], 0) / max(sum(cond[digits[i]].values()), 1)
        d10 = min(companion_score * 0.15 + pair_bonus * 2, 6)
        details['伴随'] = round(d10, 1)
        score += d10

        # ── 11. 遗漏甜蜜区（基于Z-score，满分8）──
        miss_score = 0
        for pos, digit in enumerate(digits):
            z = ctx['miss_zscore'][pos].get(digit, 0)
            if z >= 1.5:
                miss_score += min(z * 1.5, 3)
            elif z >= 0.5:
                miss_score += 0.5
            elif z < -1.0:
                # 过热的数字轻微扣分
                miss_score -= 0.5
        d11 = max(0, min(miss_score, 8))
        details['遗漏Z'] = round(d11, 1)
        score += d11

        # ── 12. 风格调整 ──
        # conservative 偏好热号，aggressive 偏好冷号，balanced 不调整
        if style == 'conservative':
            # 增加热号奖励
            heat_raw = sum(ctx['heat_weighted'][pos].get(digits[pos], 0) for pos in range(3))
            total_heat = sum(sum(h.values()) for h in ctx['heat_weighted']) or 1
            heat_ratio = heat_raw / (3 * total_heat / 10)
            if heat_ratio > 1.2: score += 2
        elif style == 'aggressive':
            # 增加冷号奖励
            cold_raw = sum(ctx['miss'][pos].get(digits[pos], 0) for pos in range(3))
            if cold_raw > 40: score += 2

        return round(score, 1), details
    
    def _weighted_digit_pick(self, pos: int, ctx: Dict, style: str = 'balanced') -> int:
        """根据位置得分加权随机选一个数字，保证随机性同时偏向高分"""
        nums = list(range(10))
        weights = []
        for digit in nums:
            base = self._position_digit_score(pos, digit, ctx, style)
            # 加入少量随机扰动，避免总是选同一数字
            noise = random.uniform(-0.1, 0.1)
            weights.append(max(0.05, base + noise))
        total = sum(weights)
        r = random.uniform(0, total)
        cum = 0
        for num, w in zip(nums, weights):
            cum += w
            if r <= cum:
                return num
        return nums[-1]

    def generate_candidates_v3(self, data: List[Dict], ctx: Dict, style: str = 'balanced', n_candidates: int = 800) -> List[Tuple]:
        candidates = set()

        # 确定目标形态：根据形态转移概率和近期形态
        recent_total = sum(ctx['recent_shapes'].values()) or 20
        zuliu_ratio = ctx['recent_shapes'].get('zuliu', int(recent_total * 0.72)) / recent_total
        target_zuliu = zuliu_ratio > 0.65  # 大多数情况下偏好组六

        def add_candidate(combo):
            combo = tuple(combo)
            # V3.2 融合：V3.0 此处硬性只收组六（len(set)!=3 直接丢弃），
            # 导致清单 100% 组六、组三（理论占27%）永远缺席。
            # 现放宽为"排除豹子、保留组三与组六"，由 select_top10 的第零轮做组三保底，
            # 主流程仍由高分候选主导。
            if len(set(combo)) < 2:
                return
            candidates.add(combo)

        # 策略1: 基于位置得分的加权采样（核心，生成约一半候选）
        for _ in range(400):
            combo = [self._weighted_digit_pick(pos, ctx, style) for pos in range(3)]
            add_candidate(combo)

        # 策略2: 贝叶斯采样（保证组六）
        for _ in range(100):
            combo = []
            for pos in range(3):
                nums = list(range(10))
                probs = [ctx['bayesian'][pos].get(d, 0.1) for d in nums]
                total_p = sum(probs) or 1
                probs = [p / total_p for p in probs]
                r = random.random()
                cum = 0
                for num, p in zip(nums, probs):
                    cum += p
                    if r <= cum:
                        combo.append(num)
                        break
            add_candidate(combo)

        # 策略3: 遗漏回补组合（基于 Z-score 选真正异常遗漏）
        high_miss = []
        for pos in range(3):
            for digit in range(10):
                z = ctx['miss_zscore'][pos].get(digit, 0)
                if z >= 0.8:
                    high_miss.append((pos, digit, z))
        high_miss.sort(key=lambda x: -x[2])
        for _ in range(150):
            combo = [self._weighted_digit_pick(pos, ctx, style) for pos in range(3)]
            if high_miss:
                # 70% 概率注入一个高遗漏数字
                if random.random() < 0.7:
                    pos, digit, _ = random.choice(high_miss[:8])
                    combo[pos] = digit
            add_candidate(combo)

        # 策略4: 和值目标采样（结合和尾遗漏）
        # 选择和尾：优先选择和尾 Z-score 高的
        tail_candidates = sorted(range(10), key=lambda t: ctx['sum_tail_zscore'].get(t, 0), reverse=True)[:5]
        for _ in range(150):
            target_tail = random.choice(tail_candidates) if tail_candidates and random.random() < 0.6 else random.randint(0, 9)
            target_hz = random.randint(7, 20) * 10 + target_tail  # 确保尾数
            # 更简单：直接生成随机组六然后看尾数是否匹配
            for _ in range(20):
                b = random.randint(0, 9)
                s = random.randint(0, 9)
                g_needed_candidates = [g for g in range(10) if (b + s + g) % 10 == target_tail and len({b, s, g}) == 3]
                if g_needed_candidates:
                    g = random.choice(g_needed_candidates)
                    add_candidate((b, s, g))
                    break

        # 策略5: 跨度目标采样（结合跨度遗漏）
        span_candidates = sorted(range(10), key=lambda sp: ctx['span_zscore'].get(sp, 0), reverse=True)[:5]
        for _ in range(120):
            target_span = random.choice(span_candidates) if span_candidates and random.random() < 0.6 else random.choice([3, 4, 5, 6, 7])
            if not (0 <= target_span <= 9): continue
            lo = random.randint(0, 9 - target_span)
            hi = lo + target_span
            pool = [x for x in range(lo, hi + 1)]
            if len(pool) >= 3:
                b, s, g = random.sample(pool, 3)
                add_candidate((b, s, g))

        # 策略6: 伴随号组合（保证组六）
        top_pairs = ctx['companion_pairs'].most_common(15)
        for _ in range(100):
            if top_pairs:
                pair = random.choice(top_pairs)[0]
                third = random.randint(0, 9)
                while third in pair:
                    third = random.randint(0, 9)
                combo = [pair[0], pair[1], third]
                random.shuffle(combo)
                add_candidate(combo)

        # 策略7: 蒙特卡洛+多维过滤（保证组六，放宽条件增加多样性）
        for _ in range(300):
            b, s, g = random.randint(0, 9), random.randint(0, 9), random.randint(0, 9)
            if len({b, s, g}) != 3:
                continue
            hz = b + s + g
            kd = max(b, s, g) - min(b, s, g)
            # 比原来更宽的范围，但排除极端
            if (5 <= hz <= 22 and 2 <= kd <= 8):
                add_candidate((b, s, g))

        return list(candidates)
    
    def select_top10_with_coverage(self, candidates: List[Tuple], ctx: Dict, style: str = 'balanced') -> List[Dict]:
        """
        选择10注号码：在评分最高的候选中保证多样性。
        关键改进：
        - 不再硬凑覆盖面，而是以高分为基础，用 soft diversity 惩罚重复属性
        - 避免同一和值、同一跨度过度集中
        - V3.1 注入：质量底线过滤、组三形态保底、组合级去重
        """
        if not candidates:
            return []

        # 评分并记录属性
        def make_scored_item(b, s, g):
            sc, details = self.score_candidate_v3(b, s, g, ctx, style)
            hz = b + s + g
            kd = max(b, s, g) - min(b, s, g)
            unique = len(set([b, s, g]))
            shape = 'baozi' if unique == 1 else ('zusan' if unique == 2 else 'zuliu')
            odd = sum(1 for x in [b, s, g] if x % 2 == 1)
            big = sum(1 for x in [b, s, g] if x >= 5)
            return {
                'score': sc, 'b': b, 's': s, 'g': g,
                'hz': hz, 'kd': kd, 'shape': shape,
                'odd': odd, 'big': big, 'details': details
            }

        scored = [make_scored_item(*c) for c in candidates]

        # 质量底线（V3.1）：剔除极端组合（跨度<3 或 >8，和值<6 或 >23）
        # 跨度4-7占56.4%、和值7-20占83.2%，清单内不应出现跨度1/9、和值26这类尾部组合。
        filtered = [it for it in scored if 3 <= it['kd'] <= 8 and 6 <= it['hz'] <= 23]
        if len(filtered) >= 10:
            scored = filtered
        # 过滤后不足10注时保留原池，避免清单凑不满

        # 按分数降序
        scored.sort(key=lambda x: -x['score'])

        selected = []
        seen = set()          # 组合级去重：sorted 元组相同 = 同一组选号码
        span_count = Counter()
        hz_count = Counter()
        odd_count = Counter()
        big_count = Counter()

        def combo_key(item):
            return tuple(sorted((item['b'], item['s'], item['g'])))

        def take(item, shape_counter=None):
            selected.append(item)
            seen.add(combo_key(item))
            span_count[item['kd']] += 1
            hz_count[item['hz']] += 1
            odd_count[item['odd']] += 1
            big_count[item['big']] += 1

        def diversity_penalty(item):
            """已选集合中相似属性越多，惩罚越大"""
            penalty = 0
            # 跨度重复惩罚
            penalty += span_count[item['kd']] * 3
            # 和值重复惩罚
            penalty += hz_count[item['hz']] * 3
            # 奇偶/大小比例重复惩罚
            penalty += odd_count[item['odd']] * 1
            penalty += big_count[item['big']] * 1
            return penalty

        # 第零轮（V3.1）：组三形态保底（组三理论占比27%），按评分降序取用，不随机
        # V3.2：取3注——金码常与最高分组三重合，保底2注会被组合级去重吃掉一注
        zusan_scored = [it for it in scored if it['shape'] == 'zusan']
        for item in zusan_scored[:3]:
            if combo_key(item) in seen:
                continue
            take(item)

        # 第一轮：高分优先 + 多样性惩罚
        # 使用贪心：每次从剩余候选中选 (score - penalty) 最高的
        remaining = [it for it in scored[:300] if combo_key(it) not in seen]
        while len(selected) < 10 and remaining:
            best_idx = 0
            best_value = remaining[0]['score'] - diversity_penalty(remaining[0])
            for idx in range(1, len(remaining)):
                val = remaining[idx]['score'] - diversity_penalty(remaining[idx])
                if val > best_value:
                    best_value = val
                    best_idx = idx
            item = remaining.pop(best_idx)
            if combo_key(item) in seen:
                continue
            take(item)

        # 第二轮：贪心池耗尽时从全量候选按评分补足（仍受组合级去重约束）
        if len(selected) < 10:
            for item in scored:
                if len(selected) >= 10:
                    break
                if combo_key(item) in seen:
                    continue
                take(item)

        return selected[:10]
    
    def generate_gold_silver(self, data: List[Dict], ctx: Dict, style: str = 'balanced') -> Tuple[Tuple, Tuple]:
        """
        生成金码和银码推荐：
        - 金码：全枚举 + 重号/邻号奖励，得分最高的号码
        - 银码：与金码数字重叠不超过1个、形态尽量不同的次高号码
        """
        if not data:
            return (1, 2, 3), (4, 5, 6)

        last_draw = (data[0]['h'], data[0]['t'], data[0]['u'])
        last_set = set(last_draw)
        neighbor_set = set()
        for d in last_draw:
            neighbor_set.update({(d - 1) % 10, (d + 1) % 10})
        neighbor_set -= last_set

        def full_enumerate(filter_style: str = 'balanced', exclude: set = None):
            candidates = []
            for b in range(10):
                for s in range(10):
                    for g in range(10):
                        combo = (b, s, g)
                        if exclude and combo in exclude:
                            continue
                        hz = b + s + g
                        kd = max(b, s, g) - min(b, s, g)
                        # 放宽枚举条件：只要和值不太极端
                        if 5 <= hz <= 22 and 2 <= kd <= 8:
                            score, details = self.score_candidate_v3(b, s, g, ctx, filter_style)
                            # 重号奖励：与上期号码有交集
                            overlap = len(set(combo) & last_set)
                            if overlap >= 1:
                                score += 1.5
                            # 邻号奖励：包含上期邻号
                            neighbor_overlap = len(set(combo) & neighbor_set)
                            if neighbor_overlap >= 1:
                                score += 1.0
                            # 金码偏好组六
                            if len(set(combo)) == 3:
                                score += 0.5
                            candidates.append((score, b, s, g, details))
            candidates.sort(key=lambda x: -x[0])
            return candidates

        # 金码：保守策略 + 重号/邻号奖励
        gold_candidates = full_enumerate('conservative')
        gold = gold_candidates[0][1:4]

        # 银码：排除金码的所有排列 + 与金码数字重叠 ≤1 个，且尽量不同形态
        # （V3.1）sorted 元组相同即视为同一组选，银码与金码同组合 = 重复押注同一组选
        gold_set = set(gold)
        gold_combo = tuple(sorted(gold))
        silver_candidates = []
        for item in full_enumerate('aggressive'):
            score, b, s, g, details = item
            if tuple(sorted((b, s, g))) == gold_combo:
                continue
            combo_set = set([b, s, g])
            overlap = len(combo_set & gold_set)
            if overlap <= 1:
                # 额外奖励：形态不同
                if len(combo_set) != len(gold_set):
                    score += 0.5
                silver_candidates.append((score, b, s, g, details))

        silver_candidates.sort(key=lambda x: -x[0])
        # 重叠≤1 的候选优先；若全被排除则退回"仅排除金码排列"的次优解
        fallback = [c for c in full_enumerate('aggressive')
                    if tuple(sorted(c[1:4])) != gold_combo]
        silver = (silver_candidates[0] if silver_candidates
                  else (fallback[0] if fallback else gold_candidates[1]))[1:4]

        return gold, silver

    def enforce_gold_silver(self, selected: List[Dict], ctx: Dict, gold: Tuple, silver: Tuple,
                            style: str = 'balanced') -> List[Dict]:
        """
        （V3.1 移植）保证金码/银码一定出现在10注清单内，并消除"同组合不同排列"造成的名额浪费。

        处理顺序：
        1. 组合级去重（sorted 元组相同 = 同一组选号码，只保留评分较高者）
        2. 金码、银码占位，保证两者一定在清单内
        3. 不足10注时从候选池（沿用质量底线）按评分补足
        4. 数字覆盖校正：0-9每个数字至少出现1次
        5. 分散度校正：单个数字在10注中出现次数不超过5（防止"7注都含同一个数字"这类隐性集中）
        """
        def combo_key(nums):
            return tuple(sorted(nums))

        def make_item(nums):
            b, s, g = nums
            sc, details = self.score_candidate_v3(b, s, g, ctx, style)
            hz = b + s + g
            kd = max(b, s, g) - min(b, s, g)
            unique = len(set([b, s, g]))
            shape = 'baozi' if unique == 1 else ('zusan' if unique == 2 else 'zuliu')
            return {
                'score': sc, 'b': b, 's': s, 'g': g, 'hz': hz, 'kd': kd, 'shape': shape,
                'odd': sum(1 for x in [b, s, g] if x % 2 == 1),
                'big': sum(1 for x in [b, s, g] if x >= 5),
                'details': details
            }

        # 全量候选池（沿用质量底线：跨度3-8、和值6-23），按评分降序
        pool = []
        for b in range(10):
            for s in range(10):
                for g in range(10):
                    hz, kd = b + s + g, max(b, s, g) - min(b, s, g)
                    if not (3 <= kd <= 8 and 6 <= hz <= 23):
                        continue
                    sc, _ = self.score_candidate_v3(b, s, g, ctx, style)
                    pool.append((sc, b, s, g))
        pool.sort(key=lambda x: -x[0])

        items = []
        used = set()

        # 1) 金码、银码优先占位（保证两者一定在清单内）
        for target in (gold, silver):
            k = combo_key(target)
            if k in used:
                continue
            items.append(make_item(tuple(target)))
            used.add(k)

        # 2) 原10注按评分补充，跳过已占用组合（消除 615/516 这类同组合浪费）
        for item in sorted(selected, key=lambda x: -x['score']):
            if len(items) >= 10:
                break
            k = combo_key((item['b'], item['s'], item['g']))
            if k in used:
                continue
            items.append(item)
            used.add(k)

        # 3) 仍不足10注则从候选池补足
        for sc, b, s, g in pool:
            if len(items) >= 10:
                break
            k = combo_key((b, s, g))
            if k in used:
                continue
            items.append(make_item((b, s, g)))
            used.add(k)

        # 4) 数字覆盖校正：0-9每个数字在10注中至少出现1次
        #    缺失则用候选池内评分最高且含该数字的号码，替换评分最低（且非金/银码）的一注
        #    V3.2 改进：优先替换"安全"的注——其每个数字都由其他注兜底，
        #    否则会出现"补了4、丢了7"的震荡，10次迭代也收敛不了。
        gold_key, silver_key = combo_key(gold), combo_key(silver)
        for _ in range(30):
            digit_count = Counter()
            digit_src = Counter()
            for it in items:
                for d in (it['b'], it['s'], it['g']):
                    digit_count[d] += 1
                for d in set((it['b'], it['s'], it['g'])):
                    digit_src[d] += 1
            missing = [d for d in range(10) if digit_count[d] == 0]
            if not missing:
                break
            target = missing[0]
            items.sort(key=lambda x: x['score'])  # 升序：首个匹配即最低分

            def key_of(it):
                return combo_key((it['b'], it['s'], it['g']))

            def is_safe(it):
                return all(digit_src[d] >= 2 for d in set((it['b'], it['s'], it['g'])))

            # 优先级：非金/银码 + 非组三 + 安全替换 > 非金/银码 + 非组三 > 任意非金/银码
            # 组三是形态覆盖的稀缺资源，不能因分数偏低被数字覆盖校正挤掉
            for predicate in (
                lambda it: key_of(it) not in (gold_key, silver_key) and it.get('shape') != 'zusan' and is_safe(it),
                lambda it: key_of(it) not in (gold_key, silver_key) and it.get('shape') != 'zusan',
                lambda it: key_of(it) not in (gold_key, silver_key),
            ):
                victim_idx = next((i for i, it in enumerate(items) if predicate(it)), None)
                if victim_idx is not None:
                    break
            if victim_idx is None:
                break
            replacement = next(((b, s, g) for sc, b, s, g in pool
                                if target in (b, s, g)
                                and combo_key((b, s, g)) not in used), None)
            if replacement is None:
                break
            victim = items.pop(victim_idx)
            used.discard(combo_key((victim['b'], victim['s'], victim['g'])))
            items.append(make_item(replacement))
            used.add(combo_key(replacement))

        # 5) 分散度校正：单个数字在10注中的出现注数不超过 max_repeat
        #    防止"7注都含8"这类隐性集中——该数字一旦不开，整张清单大概率全废
        max_repeat = 5
        for _ in range(30):
            note_cnt = Counter()
            for it in items:
                for d in set((it['b'], it['s'], it['g'])):
                    note_cnt[d] += 1
            over = [d for d in range(10) if note_cnt[d] > max_repeat]
            if not over:
                break
            hot = max(over, key=lambda d: note_cnt[d])
            items.sort(key=lambda x: x['score'])  # 升序：优先牺牲评分最低的一注

            def _key_of(it):
                return combo_key((it['b'], it['s'], it['g']))

            def _removable(it):
                ds = set((it['b'], it['s'], it['g']))
                # 必须是含 hot 的非金/银码注，且移除后不破坏 0-9 全覆盖
                if hot not in ds:
                    return False
                if _key_of(it) in (gold_key, silver_key):
                    return False
                return all(note_cnt[d] - 1 >= 1 for d in ds)

            victim_idx = None
            for predicate in (
                lambda it: _removable(it) and it.get('shape') != 'zusan',
                lambda it: _removable(it),
            ):
                victim_idx = next((i for i, it in enumerate(items) if predicate(it)), None)
                if victim_idx is not None:
                    break
            if victim_idx is None:
                break

            victim = items[victim_idx]
            victim_key = _key_of(victim)
            base = Counter(note_cnt)
            for d in set((victim['b'], victim['s'], victim['g'])):
                base[d] -= 1

            def _replaceable(nums):
                trial = Counter(base)
                for d in set(nums):
                    trial[d] += 1
                # 不能让任何数字归零；已被"稀释"到允许多少，就允许保持在该水平，
                # 否则一次只能降 1 注的情况下永远找不到合法替换品（拒绝 torpid 抖动）
                for d in range(10):
                    if trial[d] < 1:
                        return False
                    if trial[d] > max(base[d], max_repeat):
                        return False
                return True

            replacement = next(((b, s, g) for sc, b, s, g in pool
                                if combo_key((b, s, g)) not in used
                                and _replaceable((b, s, g))), None)
            if replacement is None:
                break
            items.pop(victim_idx)
            used.discard(victim_key)
            items.append(make_item(replacement))
            used.add(combo_key(replacement))

        items.sort(key=lambda x: -x['score'])
        return items[:10]

    def analyze_and_recommend(self, data: List[Dict], style: str = 'balanced', top_n: int = 10) -> Tuple[List[Dict], Dict]:
        ctx = self.compute_context(data)
        candidates = self.generate_candidates_v3(data, ctx, style=style)
        selected = self.select_top10_with_coverage(candidates, ctx, style=style)
        return selected, ctx
    
    def get_analysis_report(self, data: List[Dict], style: str = 'balanced') -> Dict:
        """生成分析报告"""
        ctx = self.compute_context(data)
        
        miss_stats = {}
        for pos, key in enumerate(['h', 't', 'u']):
            miss_stats[key] = {}
            for digit in range(10):
                miss_stats[key][digit] = ctx['miss'][pos].get(digit, 0)
        
        report = {
            'data_periods': len(data),
            'style': style,
            'version': 'V3.2',
            'method': 'Dirichlet后验+指数衰减+复隔中+Z-score遗漏+多策略共识',
            'statistics': {
                'theoretical_sum_mean': 13.5,
                'theoretical_span_mean': 4.5,
                'shape_distribution': self.SHAPE_PROBS,
                'recent_shapes': dict(ctx['recent_shapes']),
                'recent_sum_avg': ctx['hz_avg'],
                'recent_span_avg': ctx['kd_avg'],
                'miss_stats': miss_stats,
                'entropy': ctx['entropy'],
                'max_entropy': ctx['max_entropy'],
                'repeat_codes': list(ctx['repeat_codes']),
                'gap_codes': list(ctx['gap_codes']),
            },
            'hot_numbers': {
                'bai': [x[0] for x in ctx['heat_weighted'][0].most_common(3)],
                'shi': [x[0] for x in ctx['heat_weighted'][1].most_common(3)],
                'ge': [x[0] for x in ctx['heat_weighted'][2].most_common(3)]
            },
            'cold_numbers': {
                'bai': [x[0] for x in ctx['freq20'][0].most_common()[-3:]],
                'shi': [x[0] for x in ctx['freq20'][1].most_common()[-3:]],
                'ge': [x[0] for x in ctx['freq20'][2].most_common()[-3:]]
            },
            'disclaimer': '基于贝叶斯+指数衰减+复隔中+多策略共识的智能选号，不是预测。'
        }
        
        return report


if __name__ == '__main__':
    print("SmartNumberSelector V3.2")
    print("请通过 run_analysis.py 调用")
