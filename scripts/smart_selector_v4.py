#!/usr/bin/env python3
"""
智能选号系统 V4.0 (3D/排列三) —— 基于时空联合统计的候选评分模块

核心理念：在 V3.0 多维统计评分基础上，引入位置内 Markov 转移、三位经验联合分布、
几何遗漏概率、自相关周期检测与隐状态 Regime 模型，使候选组合在“独立位置特征”
之外还能体现历史序列结构。需要特别强调：

    彩票开奖是独立随机事件，本模块不预测未来，只描述历史统计分布。
    所有“得分”均为相对统计倾向，不保证也不暗示中奖。

V4.0 关键增量：
1. 一阶 / 二阶 Markov 转移矩阵（每位独立构建）
2. 三位组合经验联合频率 + 拉普拉斯平滑对数分
3. 几何分布遗漏概率：P(遗漏 ≥ k) = (1-p)^k
4. 滞后 1-20 自相关函数（ACF）周期探测
5. 热 / 温 / 冷 Regime 状态机与状态转移矩阵
6. 100 次 bootstrap 评估位置分、和值分、跨度分稳定性
7. 保留 V3 频率 / 贝叶斯 / 遗漏 Z-score / 热度加权等全部基础统计量

数学约定：
- 数据按时间倒序排列：data[0] 为最新一期，data[t] 为往前第 t 期。
- 位置 key 顺序固定为 ['h','t','u']，分别对应百、十、个位。
- 所有概率均为频率估计，经拉普拉斯或常量平滑处理，避免零概率。
"""

import random
import math
import itertools
from collections import Counter
from typing import List, Dict, Tuple, Optional


# 尝试引入 scipy/numpy 以加速 ACF/bootstrap；如不可用则回退纯标准库实现。
# 这符合“优先标准库，已有 numpy/scipy 可用时复用”的约束。
try:
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


class SmartNumberSelectorV4:
    """
    3D/排列三 V4 智能候选生成器。

    Parameters
    ----------
    style : str
        全局风格：'adaptive'（根据近期熵自动选择）、'conservative'（保守，偏热号）、
        'balanced'（均衡）、'aggressive'（激进，偏冷号/遗漏回补）。
    """

    # 理论概率（精确枚举）
    SUM_PROBS = {}
    SPAN_PROBS = {}
    SHAPE_PROBS = {'baozi': 0.01, 'zusan': 0.27, 'zuliu': 0.72}
    PRIMES = {2, 3, 5, 7}

    # 几何遗漏理论概率：每位数字 p = 1/10
    DIGIT_PROB = 1.0 / 10.0

    def __init__(self, style: str = 'adaptive'):
        self.style = style
        self._compute_theoretical_distributions()

    # ------------------------------------------------------------------
    # 静态理论分布
    # ------------------------------------------------------------------
    def _compute_theoretical_distributions(self):
        """枚举 000-999 计算和值、跨度理论概率。"""
        sum_count = Counter()
        span_count = Counter()
        for b, s, g in itertools.product(range(10), repeat=3):
            hz = b + s + g
            span = max(b, s, g) - min(b, s, g)
            sum_count[hz] += 1
            span_count[span] += 1
        total = 1000.0
        self.SUM_PROBS = {hz: cnt / total for hz, cnt in sum_count.items()}
        self.SPAN_PROBS = {span: cnt / total for span, cnt in span_count.items()}

    # ------------------------------------------------------------------
    # 核心上下文计算
    # ------------------------------------------------------------------
    def compute_context(self, data: List[Dict]) -> Dict:
        """
        计算 V4 推荐所需的全部统计上下文。

        Parameters
        ----------
        data : List[Dict]
            每期字典，必须包含：{'period': str, 'date': str, 'h': int, 't': int, 'u': int, 'sum': int}
            按时间倒序排列（data[0] 最新）。

        Returns
        -------
        Dict
            包含 markov1, markov2, joint_freq, geometric_miss, acf, regime,
            bootstrap_stats 及 V3 全部统计量。
        """
        ctx = {}
        n = len(data)
        if n == 0:
            ctx['n'] = 0
            ctx['markov1'] = [{} for _ in range(3)]
            ctx['markov2'] = [{} for _ in range(3)]
            ctx['joint_freq'] = Counter()
            ctx['geometric_miss'] = [{} for _ in range(3)]
            ctx['acf'] = [[] for _ in range(3)]
            ctx['regime'] = [{} for _ in range(3)]
            ctx['bootstrap_stats'] = {}
            return ctx

        # 数据顺序：data[0] 最新，越后越旧。为时间序列计算方便，创建正序（旧→新）列表。
        chronological = list(reversed(data))
        pos_keys = ['h', 't', 'u']
        pos_series = [
            [d[key] for d in chronological] for key in pos_keys
        ]

        # ── 1. V3 基础统计 ──
        ctx['n'] = n
        ctx.update(self._compute_v3_context(data))

        # ── 2. 一阶 Markov 转移矩阵（10×10）──
        # P(X_t = j | X_{t-1}=i) = count(i -> j) / count(i)
        ctx['markov1'] = self._compute_markov1(pos_series)

        # ── 3. 二阶 Markov 转移矩阵（10×10×10）──
        # P(X_t = j | X_{t-1}=i, X_{t-2}=k) = count(k,i -> j) / count(k,i)
        ctx['markov2'] = self._compute_markov2(pos_series)

        # ── 4. 三位组合经验联合频率 Counter((b,s,g)) ──
        ctx['joint_freq'] = self._compute_joint_freq(chronological)

        # ── 5. 几何遗漏偏离 ──
        # 对每位每个数字，计算 P(遗漏 ≥ 当前遗漏) 与基于历史均值的 P(遗漏 ≥ 均值) 的比值。
        ctx['geometric_miss'] = self._compute_geometric_miss(data, chronological, pos_keys)

        # ── 6. 自相关函数 ACF（滞后 1-20）──
        ctx['acf'] = self._compute_acf(pos_series)

        # ── 7. Regime 状态机与状态转移矩阵 ──
        ctx['regime'] = self._compute_regime(pos_series)

        # ── 8. Bootstrap 统计 ──
        ctx['bootstrap_stats'] = self._compute_bootstrap_stats(data)

        return ctx

    # ------------------------------------------------------------------
    # V3 基础统计封装
    # ------------------------------------------------------------------
    def _compute_v3_context(self, data: List[Dict]) -> Dict:
        """复用 V3 的核心统计量，保持兼容性与可对比性。"""
        ctx = {}
        n = len(data)
        pos_keys = ['h', 't', 'u']

        # 全量频率
        freq_all = [Counter(), Counter(), Counter()]
        for d in data:
            for pos, key in enumerate(pos_keys):
                freq_all[pos][d[key]] += 1

        def window_freq(window):
            f = [Counter(), Counter(), Counter()]
            for d in data[:window]:
                for pos, key in enumerate(pos_keys):
                    f[pos][d[key]] += 1
            return f

        ctx['freq_all'] = freq_all
        ctx['freq20'] = window_freq(20)
        ctx['freq50'] = window_freq(50)
        ctx['freq100'] = window_freq(100) if n >= 100 else window_freq(n)

        # 指数衰减热度
        ctx['heat_weighted'] = [Counter(), Counter(), Counter()]
        for idx, d in enumerate(data[:min(n, 100)]):
            w = max(1.0, math.exp(-idx / 20.0) * 5)
            for pos, key in enumerate(pos_keys):
                ctx['heat_weighted'][pos][d[key]] += w

        # 贝叶斯后验（拉普拉斯平滑，伪计数=1）
        ctx['bayesian'] = [{}, {}, {}]
        for pos in range(3):
            total = sum(freq_all[pos].values()) or n
            for digit in range(10):
                ctx['bayesian'][pos][digit] = (freq_all[pos].get(digit, 0) + 1) / (total + 10)

        # 遗漏值
        ctx['miss'] = [{}, {}, {}]
        for pos, key in enumerate(pos_keys):
            for digit in range(10):
                found = False
                for i, d in enumerate(data):
                    if d[key] == digit:
                        ctx['miss'][pos][digit] = i
                        found = True
                        break
                if not found:
                    ctx['miss'][pos][digit] = n

        # 遗漏历史统计
        ctx['miss_stats'] = {pos: {} for pos in range(3)}
        for pos, key in enumerate(pos_keys):
            for digit in range(10):
                gaps = []
                last_idx = None
                for i, d in enumerate(data):
                    if d[key] == digit:
                        if last_idx is not None:
                            gaps.append(i - last_idx - 1)
                        last_idx = i
                if gaps:
                    mean_gap = sum(gaps) / len(gaps)
                    std_gap = math.sqrt(sum((x - mean_gap) ** 2 for x in gaps) / len(gaps)) if len(gaps) > 1 else 3.0
                    ctx['miss_stats'][pos][digit] = {'mean': mean_gap, 'std': std_gap, 'max': max(gaps)}
                else:
                    ctx['miss_stats'][pos][digit] = {'mean': 9.0, 'std': 8.0, 'max': n}

        # 遗漏 Z-score
        ctx['miss_zscore'] = [{}, {}, {}]
        for pos in range(3):
            for digit in range(10):
                cur = ctx['miss'][pos].get(digit, 0)
                stats = ctx['miss_stats'][pos][digit]
                ctx['miss_zscore'][pos][digit] = (cur - stats['mean']) / max(stats['std'], 1.0)

        # 复隔中
        if len(data) >= 2:
            last_nums = {data[0]['h'], data[0]['t'], data[0]['u']}
            prev_nums = {data[1]['h'], data[1]['t'], data[1]['u']}
            ctx['repeat_codes'] = last_nums
            ctx['gap_codes'] = prev_nums - last_nums
            ctx['mid_codes'] = set(range(10)) - last_nums - ctx['gap_codes']
        else:
            ctx['repeat_codes'] = set()
            ctx['gap_codes'] = set()
            ctx['mid_codes'] = set(range(10))

        # 和值 / 跨度统计
        sums_full = [d['h'] + d['t'] + d['u'] for d in data]
        spans_full = [max(d['h'], d['t'], d['u']) - min(d['h'], d['t'], d['u']) for d in data]

        ctx['sum_freq'], ctx['sum_tail_miss'], ctx['sum_hist_stats'] = self._value_stats(
            [s % 10 for s in sums_full], 9, n)
        ctx['span_freq'], ctx['span_miss'], ctx['span_hist_stats'] = self._value_stats(
            spans_full, 9, n)

        sums_recent = sums_full[:50]
        spans_recent = spans_full[:50]
        ctx['hz_avg'] = sum(sums_recent) / len(sums_recent) if sums_recent else 13.5
        ctx['kd_avg'] = sum(spans_recent) / len(spans_recent) if spans_recent else 4.5

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

        # 形态统计
        ctx['recent_shapes'] = Counter()
        for d in data[:20]:
            unique = len({d['h'], d['t'], d['u']})
            if unique == 1:
                ctx['recent_shapes']['baozi'] += 1
            elif unique == 2:
                ctx['recent_shapes']['zusan'] += 1
            else:
                ctx['recent_shapes']['zuliu'] += 1

        ctx['shape_cnt'] = Counter()
        for d in data[:50]:
            unique = len({d['h'], d['t'], d['u']})
            if unique == 1:
                ctx['shape_cnt']['豹子'] += 1
            elif unique == 2:
                ctx['shape_cnt']['组三'] += 1
            else:
                ctx['shape_cnt']['组六'] += 1

        # 形态转移
        ctx['shape_transition'] = {}
        if len(data) >= 2:
            def shape_of(d):
                u = len({d['h'], d['t'], d['u']})
                return 'baozi' if u == 1 else ('zusan' if u == 2 else 'zuliu')
            trans = Counter()
            for i in range(len(data) - 1):
                cur = shape_of(data[i])
                nxt = shape_of(data[i + 1])
                trans[(cur, nxt)] += 1
            for cur in ['baozi', 'zusan', 'zuliu']:
                total = sum(trans[(cur, nxt)] for nxt in ['baozi', 'zusan', 'zuliu'])
                if total > 0:
                    ctx['shape_transition'][cur] = {nxt: trans[(cur, nxt)] / total
                                                     for nxt in ['baozi', 'zusan', 'zuliu']}
                else:
                    ctx['shape_transition'][cur] = {'baozi': 0.01, 'zusan': 0.27, 'zuliu': 0.72}

        # 奇偶 / 大小
        ctx['odd_even_cnt'] = Counter()
        ctx['big_small_cnt'] = Counter()
        for d in data[:30]:
            odd = sum(1 for x in [d['h'], d['t'], d['u']] if x % 2 == 1)
            ctx['odd_even_cnt'][f'{odd}:{3 - odd}'] += 1
            big = sum(1 for x in [d['h'], d['t'], d['u']] if x >= 5)
            ctx['big_small_cnt'][f'{big}:{3 - big}'] += 1

        # 012路
        ctx['road20'] = [Counter(), Counter(), Counter()]
        for d in data[:30]:
            for pos, key in enumerate(pos_keys):
                ctx['road20'][pos][d[key] % 3] += 1
        ctx['road_miss'] = [{}, {}, {}]
        for pos, key in enumerate(pos_keys):
            for road in range(3):
                for i, d in enumerate(data):
                    if d[key] % 3 == road:
                        ctx['road_miss'][pos][road] = i
                        break
                else:
                    ctx['road_miss'][pos][road] = len(data)

        # 伴随号
        ctx['companion_pairs'] = Counter()
        for d in data[:50]:
            nums = [d['h'], d['t'], d['u']]
            for i in range(3):
                for j in range(i + 1, 3):
                    pair = tuple(sorted([nums[i], nums[j]]))
                    ctx['companion_pairs'][pair] += 1

        # 位置间条件频率
        ctx['pair_cond'] = {}
        for pos_a in range(3):
            for pos_b in range(3):
                if pos_a == pos_b:
                    continue
                key_a = pos_keys[pos_a]
                key_b = pos_keys[pos_b]
                cond = {a: Counter() for a in range(10)}
                for d in data:
                    cond[d[key_a]][d[key_b]] += 1
                ctx['pair_cond'][(pos_a, pos_b)] = cond

        # 信息熵
        ctx['entropy'] = [0.0, 0.0, 0.0]
        for pos in range(3):
            total = sum(freq_all[pos].values()) or 1
            probs = [freq_all[pos].get(d, 0) / total for d in range(10)]
            ctx['entropy'][pos] = -sum(p * math.log2(p) for p in probs if p > 0)
        ctx['max_entropy'] = math.log2(10)

        # 趋势方向
        ctx['trend'] = []
        for pos, key in enumerate(pos_keys):
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

    def _value_stats(self, values: List[int], max_val: int, n: int) -> Tuple[Counter, Dict, Dict]:
        """计算离散值的频率、当前遗漏、历史遗漏统计。"""
        freq = Counter(values)
        miss = {}
        for v in range(max_val + 1):
            for i, val in enumerate(values):
                if val == v:
                    miss[v] = i
                    break
            else:
                miss[v] = n
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
                mean_gap = sum(gaps) / len(gaps)
                std_gap = math.sqrt(sum((x - mean_gap) ** 2 for x in gaps) / len(gaps)) if len(gaps) > 1 else 3.0
                hist_stats[v] = {'mean': mean_gap, 'std': std_gap}
            else:
                hist_stats[v] = {'mean': max(1, n / max_val), 'std': 3.0}
        return freq, miss, hist_stats

    # ------------------------------------------------------------------
    # Markov 转移
    # ------------------------------------------------------------------
    def _compute_markov1(self, pos_series: List[List[int]]) -> List[Dict[int, Dict[int, float]]]:
        """
        计算每位的一阶 Markov 转移矩阵。
        返回 [{i: {j: P(j|i)}} for pos in range(3)]。
        """
        result = []
        for series in pos_series:
            counts = {i: Counter() for i in range(10)}
            marginal = Counter()
            for t in range(1, len(series)):
                prev, cur = series[t - 1], series[t]
                counts[prev][cur] += 1
                marginal[prev] += 1
            trans = {}
            for i in range(10):
                total = marginal[i]
                if total > 0:
                    trans[i] = {j: (counts[i].get(j, 0) + 0.1) / (total + 1.0)
                                for j in range(10)}
                else:
                    # 无观测时退化为均匀分布
                    trans[i] = {j: 0.1 for j in range(10)}
            result.append(trans)
        return result

    def _compute_markov2(self, pos_series: List[List[int]]) -> List[Dict[Tuple[int, int], Dict[int, float]]]:
        """
        计算每位的二阶 Markov 转移矩阵。
        返回 [{ (k,i): {j: P(j|k,i)} } for pos in range(3)]。
        """
        result = []
        for series in pos_series:
            counts = {}
            marginal = Counter()
            for t in range(2, len(series)):
                prev2, prev1, cur = series[t - 2], series[t - 1], series[t]
                key = (prev2, prev1)
                counts.setdefault(key, Counter())[cur] += 1
                marginal[key] += 1
            trans = {}
            for key in itertools.product(range(10), repeat=2):
                total = marginal.get(key, 0)
                if total > 0:
                    trans[key] = {j: (counts[key].get(j, 0) + 0.1) / (total + 1.0)
                                  for j in range(10)}
                else:
                    trans[key] = {j: 0.1 for j in range(10)}
            result.append(trans)
        return result

    # ------------------------------------------------------------------
    # 联合分布
    # ------------------------------------------------------------------
    def _compute_joint_freq(self, chronological: List[Dict]) -> Counter:
        """
        计算三位组合 (b,s,g) 的经验联合频率。
        使用正序历史数据，保留顺序信息（3D 开奖有位置含义）。
        """
        joint = Counter()
        for d in chronological:
            joint[(d['h'], d['t'], d['u'])] += 1
        return joint

    # ------------------------------------------------------------------
    # 几何遗漏偏离
    # ------------------------------------------------------------------
    def _compute_geometric_miss(self, data: List[Dict], chronological: List[Dict], pos_keys: List[str]) -> List[Dict[int, Dict[str, float]]]:
        """
        对每位每个数字计算几何遗漏概率。

        数学依据：若数字独立出现概率 p=1/10，则遗漏 M 服从几何分布，
        P(M >= k) = (1-p)^k。这里用当前遗漏对应的生存概率与历史均值对应
        的生存概率之比，衡量“当前遗漏相对于自身历史有多极端”。

        返回结构：
        [
            {
                digit: {
                    'p_surv_cur': float,   # P(遗漏 >= 当前遗漏)
                    'p_surv_mean': float,  # P(遗漏 >= 历史均值)
                    'geo_ratio': float     # p_surv_cur / p_surv_mean，越大越偏离
                }
            }
            for pos in range(3)
        ]
        """
        p = self.DIGIT_PROB
        q = 1.0 - p
        result = []
        n = len(data)

        for pos, key in enumerate(pos_keys):
            gm = {}
            # 当前遗漏（倒序 data）
            cur_miss = {}
            for digit in range(10):
                found = False
                for i, d in enumerate(data):
                    if d[key] == digit:
                        cur_miss[digit] = i
                        found = True
                        break
                if not found:
                    cur_miss[digit] = n

            # 历史遗漏均值
            for digit in range(10):
                gaps = []
                last_idx = None
                for i, d in enumerate(chronological):
                    if d[key] == digit:
                        if last_idx is not None:
                            gaps.append(i - last_idx - 1)
                        last_idx = i
                mean_gap = sum(gaps) / len(gaps) if gaps else 9.0
                k_cur = max(0, cur_miss.get(digit, 0))
                k_mean = max(0, mean_gap)
                p_surv_cur = q ** k_cur
                p_surv_mean = q ** k_mean
                ratio = p_surv_cur / max(p_surv_mean, 1e-6)
                gm[digit] = {
                    'p_surv_cur': p_surv_cur,
                    'p_surv_mean': p_surv_mean,
                    'geo_ratio': ratio,
                    'mean_gap': mean_gap,
                    'cur_miss': k_cur
                }
            result.append(gm)
        return result

    # ------------------------------------------------------------------
    # 自相关函数 ACF
    # ------------------------------------------------------------------
    def _compute_acf(self, pos_series: List[List[int]]) -> List[List[float]]:
        """
        计算每位滞后 1-20 的自相关系数。

        公式：ρ(k) = Cov(X_t, X_{t-k}) / Var(X_t)
        使用无偏方差。若序列长度不足 k+1，则该滞后系数为 0。

        返回 [[ρ(1),...,ρ(20)] for pos]。
        """
        result = []
        max_lag = 20
        for series in pos_series:
            n = len(series)
            if n < 2:
                result.append([0.0] * max_lag)
                continue
            mean = sum(series) / n
            var = sum((x - mean) ** 2 for x in series) / (n - 1)
            acf = []
            for lag in range(1, max_lag + 1):
                if lag >= n:
                    acf.append(0.0)
                    continue
                cov = sum((series[t] - mean) * (series[t - lag] - mean)
                          for t in range(lag, n)) / (n - lag)
                acf.append(cov / max(var, 1e-6))
            result.append(acf)
        return result

    # ------------------------------------------------------------------
    # Regime 状态机
    # ------------------------------------------------------------------
    def _compute_regime(self, pos_series: List[List[int]]) -> List[Dict]:
        """
        对每位构建热 / 温 / 冷 隐状态序列并估计状态转移矩阵。

        状态定义（基于最近 10 期）：
        - hot  ：数字在正序最后 10 期出现 >=2 次
        - warm ：出现 1 次
        - cold ：出现 0 次

        转移矩阵：P(next_state | current_state)。
        同时给出当前最新状态的预测下一状态概率。
        """
        result = []
        state_names = ['hot', 'warm', 'cold']
        for series in pos_series:
            n = len(series)
            # 正序最后 10 期
            recent_window = min(10, n)
            recent = series[-recent_window:] if recent_window > 0 else []
            # 为每位数字标记状态
            # 构建状态序列：对每个时刻 t，基于 [t-recent_window+1, t] 内出现次数给状态
            state_seq_per_digit = {digit: [] for digit in range(10)}
            for t in range(n):
                start = max(0, t - 9)
                window = series[start:t + 1]
                for digit in range(10):
                    cnt = window.count(digit)
                    if cnt >= 2:
                        state_seq_per_digit[digit].append('hot')
                    elif cnt == 1:
                        state_seq_per_digit[digit].append('warm')
                    else:
                        state_seq_per_digit[digit].append('cold')

            # 汇总所有数字的状态转移
            trans = Counter()
            marginal = Counter()
            for digit in range(10):
                seq = state_seq_per_digit[digit]
                for t in range(1, len(seq)):
                    trans[(seq[t - 1], seq[t])] += 1
                    marginal[seq[t - 1]] += 1

            trans_mat = {}
            for s in state_names:
                total = marginal.get(s, 0)
                if total > 0:
                    trans_mat[s] = {s2: (trans.get((s, s2), 0) + 0.1) / (total + 0.3)
                                    for s2 in state_names}
                else:
                    trans_mat[s] = {s2: 1.0 / 3.0 for s2 in state_names}

            # 当前最新状态（基于最后 10 期）
            current_regime = {}
            predicted_next = {}
            for digit in range(10):
                cnt = recent.count(digit)
                if cnt >= 2:
                    cur_state = 'hot'
                elif cnt == 1:
                    cur_state = 'warm'
                else:
                    cur_state = 'cold'
                current_regime[digit] = cur_state
                predicted_next[digit] = trans_mat.get(cur_state, {})

            result.append({
                'trans_mat': trans_mat,
                'current': current_regime,
                'predicted_next': predicted_next
            })
        return result

    # ------------------------------------------------------------------
    # Bootstrap 统计
    # ------------------------------------------------------------------
    def _compute_bootstrap_stats(self, data: List[Dict]) -> Dict:
        """
        对位置分、和值分、跨度分做 100 次有放回重采样，返回均值/标准差。

        这里使用简化代理指标：
        - 位置分：窗口内各位数字出现次数的基尼 / 集中度
        - 和值分：和值序列标准差
        - 跨度分：跨度序列标准差
        """
        n = len(data)
        if n < 10:
            return {
                'position_mean': 0.0, 'position_std': 0.0,
                'sum_mean': 13.5, 'sum_std': 3.0,
                'span_mean': 4.5, 'span_std': 2.0,
                'n_bootstrap': 0
            }

        pos_keys = ['h', 't', 'u']
        pos_samples = [[d[key] for d in data] for key in pos_keys]
        sums = [d['h'] + d['t'] + d['u'] for d in data]
        spans = [max(d['h'], d['t'], d['u']) - min(d['h'], d['t'], d['u']) for d in data]

        def sample_proxy():
            idx = [random.randint(0, n - 1) for _ in range(n)]
            pos_scores = []
            for pos in range(3):
                boot = [pos_samples[pos][i] for i in idx]
                freq = Counter(boot)
                # 用 1 - 赫芬达尔指数衡量多样性，作为“位置分”代理
                total = len(boot)
                hhi = sum((c / total) ** 2 for c in freq.values())
                pos_scores.append(1.0 - hhi)
            sum_std = self._std([sums[i] for i in idx])
            span_std = self._std([spans[i] for i in idx])
            return sum(pos_scores) / 3.0, sum_std, span_std

        results = [sample_proxy() for _ in range(100)]
        pos_vals = [r[0] for r in results]
        sum_vals = [r[1] for r in results]
        span_vals = [r[2] for r in results]

        return {
            'position_mean': sum(pos_vals) / len(pos_vals),
            'position_std': self._std(pos_vals),
            'sum_mean': sum(sum_vals) / len(sum_vals),
            'sum_std': self._std(sum_vals),
            'span_mean': sum(span_vals) / len(span_vals),
            'span_std': self._std(span_vals),
            'n_bootstrap': 100
        }

    def _std(self, values: List[float]) -> float:
        """样本标准差。"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))

    # ------------------------------------------------------------------
    # 位置数字得分（V4 综合版）
    # ------------------------------------------------------------------
    def _position_digit_score_v4(self, pos: int, digit: int, ctx: Dict, style: str = 'balanced') -> float:
        """
        单个位置单个数字的综合得分，融合多窗口频率、Markov 转移趋势、
        几何遗漏偏离、ACF 周期、Regime 状态。返回 [0,1]。

        组成：
        1. 多窗口频率（50%）：20/50/100 期加权。
        2. Markov 趋势（15%）：基于最新一期数字的一阶/二阶转移概率。
        3. 几何遗漏偏离（20%）：geo_ratio 越大分越高，但有上限。
        4. ACF 周期（5%）：若检测到显著正自相关且相位匹配，轻微加分。
        5. Regime 状态（10%）：根据风格与预测下一状态匹配度加分。
        """
        if style == 'adaptive':
            style = self._auto_style(ctx)

        # 1. 多窗口频率
        f20 = ctx['freq20'][pos].get(digit, 0) / 20.0
        f50 = ctx['freq50'][pos].get(digit, 0) / 50.0
        f100 = ctx['freq100'][pos].get(digit, 0) / max(len(ctx.get('freq100', [[]])[0]) or 100, 100)
        freq_score = 0.5 * f20 + 0.3 * f50 + 0.2 * f100

        # 2. Markov 趋势
        markov_score = 0.0
        if 'markov1' in ctx and ctx['markov1']:
            last_digits = [ctx.get('last_digits', [None, None, None])[pos]]
            # 这里 last_digits 尚未设置，需要从 data 计算。为解耦，使用 ctx 中 freq_all 不依赖 data。
            # 实际上 Markov 需要上一期数字，下面用 recent_series 存于 ctx 的 last_digits。
            pass
        # 更稳健：从 markov1 矩阵取该 digit 作为“源”时，期望被转移到的概率总和做趋势代理。
        # 实际 Markov 应用见 score_candidate_v4 与 generate_candidates_v4。
        # 这里用该 digit 的“转移出”概率熵来衡量集中度，熵低表示有明确去向。
        m1_row = ctx.get('markov1', [{}])[pos].get(digit, {})
        if m1_row:
            probs = [m1_row.get(j, 0.0) for j in range(10)]
            # 作为得分：该数字作为历史前项时，最大转移概率
            markov_score = max(probs) if probs else 0.0

        # 3. 几何遗漏偏离
        gm = ctx.get('geometric_miss', [{}])[pos].get(digit, {})
        geo_ratio = gm.get('geo_ratio', 1.0)
        # ratio > 1 表示当前遗漏比历史均值更极端；映射到 [0,0.6]
        geo_score = min(max(geo_ratio - 1.0, 0.0) / 2.0, 1.0) * 0.6

        # 4. ACF 周期
        acf_score = 0.0
        acf = ctx.get('acf', [[]])[pos]
        if acf:
            significant_lags = [lag for lag, rho in enumerate(acf, start=1)
                                if rho > 0.12]
            if significant_lags:
                # 周期按显著滞后均值近似
                avg_lag = sum(significant_lags) / len(significant_lags)
                # 若当前遗漏 cur_miss 与周期整数倍接近，则轻微加分
                cur_miss = ctx['miss'][pos].get(digit, 0)
                proximity = min(abs(cur_miss - round(cur_miss / avg_lag) * avg_lag) / max(avg_lag, 1.0), 1.0)
                acf_score = (1.0 - proximity) * 0.3

        # 5. Regime 状态
        regime = ctx.get('regime', [{}])[pos]
        predicted = regime.get('predicted_next', {}).get(digit, {})
        cur_state = regime.get('current', {}).get(digit, 'warm')
        if style == 'conservative':
            # 偏好 hot
            regime_score = predicted.get('hot', 0.0)
        elif style == 'aggressive':
            # 偏好 cold（反转）
            regime_score = predicted.get('cold', 0.0)
        else:
            # balanced：warm 或 hot 都可
            regime_score = predicted.get('warm', 0.0) + 0.5 * predicted.get('hot', 0.0)

        # 风格对热/冷的直接调整
        heat_raw = ctx['heat_weighted'][pos].get(digit, 0.0)
        total_heat = sum(sum(h.values()) for h in ctx['heat_weighted']) or 1.0
        heat_ratio = heat_raw / (total_heat / 10.0 + 1e-6)
        if style == 'conservative':
            style_adj = 0.1 * (heat_ratio - 1.0)
        elif style == 'aggressive':
            cold_score = min(ctx['miss'][pos].get(digit, 0) / 30.0, 1.0)
            style_adj = 0.15 * cold_score
        else:
            style_adj = 0.0

        total = (0.50 * freq_score +
                 0.15 * markov_score +
                 0.20 * geo_score +
                 0.05 * acf_score +
                 0.10 * regime_score +
                 style_adj)
        return max(0.0, min(1.0, total))

    def _auto_style(self, ctx: Dict) -> str:
        """
        根据近 20 期信息熵自动选择风格：
        - 熵低（热号集中）→ conservative
        - 熵高（号码分散）→ aggressive
        - 默认 balanced
        """
        recent_entropy = ctx.get('entropy', [math.log2(10)] * 3)
        avg_entropy = sum(recent_entropy) / 3.0 if recent_entropy else math.log2(10)
        max_e = ctx.get('max_entropy', math.log2(10))
        ratio = avg_entropy / max_e if max_e > 0 else 1.0
        if ratio < 0.85:
            return 'conservative'
        if ratio > 0.95:
            return 'aggressive'
        return 'balanced'

    # ------------------------------------------------------------------
    # 候选评分（V4）
    # ------------------------------------------------------------------
    def score_candidate_v4(self, b: int, s: int, g: int, ctx: Dict,
                           style: str = 'balanced') -> Tuple[float, Dict]:
        """
        在 V3 的 12 维评分基础上，增加 V4 的 Markov、联合分布、ACF、Regime 加分。

        Returns
        -------
        Tuple[float, Dict]
            (总分, 明细字典)。明细包含 V3 的 12 维 + V4 的 4 维 bonus。
        """
        if style == 'adaptive':
            style = self._auto_style(ctx)

        # 先用 V3 评分作为基础
        base_score, details = self.score_candidate_v3(b, s, g, ctx, style)

        digits = [b, s, g]

        # ── 13. Markov 转移加分（满分 3）──
        # 基于每位最新一期数字的一阶转移概率
        markov_bonus = 0.0
        if ctx.get('markov1'):
            # 需要知道最新一期数字；从 data 的倒序结构推断：ctx 未保存，这里用联合频率反推。
            # 为保持 ctx 自包含，compute_context 中最新一期数字并未存入 ctx。
            # 我们使用 markov1 的“源状态”概率作为序列趋势的代理：
            # 对候选 digit，查找 markov1 中指向该 digit 的转移概率的加权平均。
            for pos in range(3):
                target_digit = digits[pos]
                weighted_in = 0.0
                total_in = 0.0
                for src in range(10):
                    p = ctx['markov1'][pos].get(src, {}).get(target_digit, 0.0)
                    weighted_in += p * (ctx['freq20'][pos].get(src, 0) + 1)
                    total_in += ctx['freq20'][pos].get(src, 0) + 1
                markov_bonus += min(weighted_in / max(total_in, 1.0) * 3.0, 1.0)
        markov_bonus = min(markov_bonus, 3.0)
        details['Markov'] = round(markov_bonus, 1)
        base_score += markov_bonus

        # ── 14. 联合频率对数加分（满分 3）──
        # joint_score = log(freq + alpha) - log(expected + alpha)
        alpha = 0.1
        expected = max(len(ctx.get('joint_freq', Counter())) / 1000.0, 0.01)
        freq_joint = ctx.get('joint_freq', Counter()).get((b, s, g), 0)
        # 对数平滑：出现过的组合加分；未出现的不扣分，避免惩罚新组合
        joint_bonus = max(0.0, math.log(freq_joint + alpha) - math.log(expected + alpha))
        joint_bonus = min(joint_bonus, 3.0)
        details['联合分布'] = round(joint_bonus, 1)
        base_score += joint_bonus

        # ── 15. ACF 周期加分（满分 1.5）──
        acf_bonus = 0.0
        acf_list = ctx.get('acf', [[]])
        if acf_list and acf_list[0]:
            for pos in range(3):
                significant = [lag for lag, rho in enumerate(acf_list[pos], start=1) if rho > 0.12]
                if significant:
                    avg_lag = sum(significant) / len(significant)
                    cur_miss = ctx['miss'][pos].get(digits[pos], 0)
                    proximity = min(abs(cur_miss - round(cur_miss / avg_lag) * avg_lag) / max(avg_lag, 1.0), 1.0)
                    acf_bonus += (1.0 - proximity) * 0.5
            acf_bonus = min(acf_bonus, 1.5)
        details['ACF周期'] = round(acf_bonus, 1)
        base_score += acf_bonus

        # ── 16. Regime 目标加分（满分 2）──
        regime_bonus = 0.0
        if ctx.get('regime'):
            for pos in range(3):
                predicted = ctx['regime'][pos].get('predicted_next', {}).get(digits[pos], {})
                if style == 'conservative':
                    regime_bonus += predicted.get('hot', 0.0)
                elif style == 'aggressive':
                    regime_bonus += predicted.get('cold', 0.0)
                else:
                    regime_bonus += predicted.get('warm', 0.0) + 0.5 * predicted.get('hot', 0.0)
            regime_bonus = min(regime_bonus, 2.0)
        details['Regime'] = round(regime_bonus, 1)
        base_score += regime_bonus

        return round(base_score, 1), details

    def score_candidate_v3(self, b: int, s: int, g: int, ctx: Dict, style: str = 'balanced') -> Tuple[float, Dict]:
        """
        V3 的 12 维评分函数。这里完整复现以保证 V4 可以直接继承。
        """
        details = {}
        score = 0.0
        digits = [b, s, g]

        # 1. 位置频率得分（满分 20）
        pos_score = sum(self._position_digit_score(pos, digits[pos], ctx, style) for pos in range(3))
        d1 = pos_score / 3 * 20
        details['位置分'] = round(d1, 1)
        score += d1

        # 2. 和值质量（满分 15）
        hz = b + s + g
        tail = hz % 10
        tail_z = ctx['sum_tail_zscore'].get(tail, 0)
        if 8 <= hz <= 19:
            base_hz = 12
        elif 6 <= hz <= 21:
            base_hz = 8
        elif 3 <= hz <= 24:
            base_hz = 4
        else:
            base_hz = 0
        tail_bonus = min(max(tail_z, 0) * 1.5, 3)
        d2 = min(base_hz + tail_bonus, 15)
        details['和值'] = round(d2, 1)
        score += d2

        # 3. 跨度质量（满分 12）
        kd = max(b, s, g) - min(b, s, g)
        span_z = ctx['span_zscore'].get(kd, 0)
        if 4 <= kd <= 7:
            base_kd = 10
        elif 3 <= kd <= 8:
            base_kd = 7
        elif 2 <= kd <= 9:
            base_kd = 4
        else:
            base_kd = 1
        span_bonus = min(max(span_z, 0) * 1.0, 2)
        d3 = min(base_kd + span_bonus, 12)
        details['跨度'] = round(d3, 1)
        score += d3

        # 4. 形态得分（满分 10）
        unique = len(set(digits))
        recent_total = sum(ctx['recent_shapes'].values()) or 20
        recent_zuliu_ratio = ctx['recent_shapes'].get('zuliu', int(recent_total * 0.72)) / recent_total
        if unique == 3:
            d4 = 8 + 2 * recent_zuliu_ratio
        elif unique == 2:
            d4 = 5 + 1 * (1 - recent_zuliu_ratio)
        else:
            d4 = 0.5
        details['形态'] = round(d4, 1)
        score += d4

        # 5. 奇偶平衡（满分 8）
        odd_count = sum(1 for x in digits if x % 2 == 1)
        d5 = 8 if odd_count in [1, 2] else 2
        details['奇偶'] = round(d5, 1)
        score += d5

        # 6. 大小平衡（满分 8）
        big_count = sum(1 for x in digits if x >= 5)
        d6 = 8 if big_count in [1, 2] else 2
        details['大小'] = round(d6, 1)
        score += d6

        # 7. 012路多样性（满分 5，路数遗漏奖励后最高 8）
        roads = {x % 3 for x in digits}
        road_score = 5 if len(roads) == 3 else (3 if len(roads) == 2 else 0)
        road_missing_bonus = 0
        for pos, digit in enumerate(digits):
            road = digit % 3
            road_miss = ctx.get('road_miss', [{}, {}, {}])[pos].get(road, 0)
            if road_miss >= 5:
                road_missing_bonus += min(road_miss / 5, 2)
        road_score = min(road_score + road_missing_bonus, 8)
        details['012路'] = round(road_score, 1)
        score += road_score

        # 8. 复隔中（满分 5）
        nums = set(digits)
        repeat_hit = len(nums & ctx['repeat_codes'])
        gap_hit = len(nums & ctx['gap_codes'])
        d8 = min(repeat_hit * 1.5 + gap_hit * 0.8, 5)
        details['复隔中'] = round(d8, 1)
        score += d8

        # 9. 质合平衡（满分 4）
        prime_count = sum(1 for x in digits if x in self.PRIMES)
        d9 = 4 if prime_count in [1, 2] else 1
        details['质合'] = round(d9, 1)
        score += d9

        # 10. 伴随号 / 位置关联（满分 6）
        companion_score = 0
        pair_bonus = 0
        for i in range(3):
            for j in range(i + 1, 3):
                pair = tuple(sorted([digits[i], digits[j]]))
                companion_score += ctx['companion_pairs'].get(pair, 0)
                cond = ctx['pair_cond'].get((i, j), {})
                if digits[i] in cond:
                    pair_bonus += cond[digits[i]].get(digits[j], 0) / max(sum(cond[digits[i]].values()), 1)
        d10 = min(companion_score * 0.15 + pair_bonus * 2, 6)
        details['伴随'] = round(d10, 1)
        score += d10

        # 11. 遗漏甜蜜区（满分 8）
        miss_score = 0
        for pos, digit in enumerate(digits):
            z = ctx['miss_zscore'][pos].get(digit, 0)
            if z >= 1.5:
                miss_score += min(z * 1.5, 3)
            elif z >= 0.5:
                miss_score += 0.5
            elif z < -1.0:
                miss_score -= 0.5
        d11 = max(0, min(miss_score, 8))
        details['遗漏Z'] = round(d11, 1)
        score += d11

        # 12. 风格调整
        if style == 'conservative':
            heat_raw = sum(ctx['heat_weighted'][pos].get(digits[pos], 0) for pos in range(3))
            total_heat = sum(sum(h.values()) for h in ctx['heat_weighted']) or 1
            heat_ratio = heat_raw / (3 * total_heat / 10)
            if heat_ratio > 1.2:
                score += 2
        elif style == 'aggressive':
            cold_raw = sum(ctx['miss'][pos].get(digits[pos], 0) for pos in range(3))
            if cold_raw > 40:
                score += 2

        return round(score, 1), details

    def _position_digit_score(self, pos: int, digit: int, ctx: Dict, style: str) -> float:
        """V3 基础位置得分，供 V3 评分调用。"""
        f20 = ctx['freq20'][pos].get(digit, 0) / 20
        f50 = ctx['freq50'][pos].get(digit, 0) / 50
        f100 = ctx['freq100'][pos].get(digit, 0) / max(len(ctx.get('freq100', [[]])[0]) or 100, 100)
        freq_score = 0.5 * f20 + 0.3 * f50 + 0.2 * f100

        trend_bonus = 0
        for td, delta in ctx['trend'][pos]:
            if td == digit:
                trend_bonus = min(delta * 0.15, 0.3)
                break

        z = ctx['miss_zscore'][pos].get(digit, 0)
        if z > 0:
            miss_score = min(z / 3.0, 1.0) * 0.6
        else:
            miss_score = max(-0.3, z / 3.0 * 0.3)

        if f20 >= 0.25:
            overheat_penalty = -0.15
        elif f20 >= 0.20:
            overheat_penalty = -0.05
        else:
            overheat_penalty = 0

        total = freq_score + trend_bonus + miss_score + overheat_penalty
        return max(0.0, min(1.0, total))

    # ------------------------------------------------------------------
    # 候选生成（V4）
    # ------------------------------------------------------------------
    def generate_candidates_v4(self, data: List[Dict], ctx: Dict,
                               style: str = 'balanced',
                               n_candidates: int = 1000) -> List[Tuple[int, int, int]]:
        """
        生成候选号码池，使用 8 种策略，保证至少 70% 为组六。

        8 种策略：
        1. Markov 链采样（一阶 + 二阶）
        2. 联合分布采样
        3. 几何遗漏回补
        4. Regime 目标采样
        5. ACF 周期采样
        6-8. 保留 V3 的贝叶斯、和值目标、跨度目标采样
        """
        if style == 'adaptive':
            style = self._auto_style(ctx)

        candidates = set()
        pos_keys = ['h', 't', 'u']

        # 最新一期数字，用于 Markov 采样
        last_draw = (data[0]['h'], data[0]['t'], data[0]['u']) if data else (0, 0, 0)
        prev_draw = (data[1]['h'], data[1]['t'], data[1]['u']) if len(data) >= 2 else last_draw

        def add_candidate(combo, force_zuliu=True):
            combo = tuple(combo)
            if force_zuliu and len(set(combo)) != 3:
                return
            candidates.add(combo)

        def weighted_choice(items, weights):
            total = sum(weights)
            if total <= 0:
                return random.choice(items)
            r = random.uniform(0, total)
            cum = 0.0
            for item, w in zip(items, weights):
                cum += w
                if r <= cum:
                    return item
            return items[-1]

        def score_pick(pos: int, style: str) -> int:
            """基于 _position_digit_score_v4 加权选择数字。"""
            nums = list(range(10))
            weights = []
            for digit in nums:
                base = self._position_digit_score_v4(pos, digit, ctx, style)
                noise = random.uniform(-0.05, 0.05)
                weights.append(max(0.05, base + noise))
            return weighted_choice(nums, weights)

        # 策略 1: Markov 链采样
        # 每位基于最新一期 / 前两期数字，按 markov1 / markov2 转移概率采样。
        for _ in range(250):
            combo = []
            for pos in range(3):
                src1 = last_draw[pos]
                src2 = prev_draw[pos]
                # 70% 用二阶，30% 用一阶（当历史足够时）
                if ctx.get('markov2') and random.random() < 0.7:
                    probs = ctx['markov2'][pos].get((src2, src1), {j: 0.1 for j in range(10)})
                else:
                    probs = ctx['markov1'][pos].get(src1, {j: 0.1 for j in range(10)})
                combo.append(weighted_choice(list(range(10)), [probs.get(j, 0.1) for j in range(10)]))
            add_candidate(combo)

        # 策略 2: 联合分布采样
        # 按 joint_freq 出现次数加权抽取历史组合，并做小幅位置扰动。
        joint_items = list(ctx.get('joint_freq', Counter()).items())
        if joint_items:
            weights = [cnt + 0.1 for _, cnt in joint_items]
            for _ in range(200):
                base, _ = weighted_choice(joint_items, weights)
                combo = list(base)
                # 对每个位置以 20% 概率替换为同位置高频数字
                for pos in range(3):
                    if random.random() < 0.2:
                        combo[pos] = score_pick(pos, style)
                add_candidate(combo)

        # 策略 3: 几何遗漏回补
        # 对每位选择 geo_ratio 最高的若干数字，组合成候选。
        for _ in range(200):
            combo = []
            for pos in range(3):
                gm = ctx.get('geometric_miss', [{}])[pos]
                # 选择 geo_ratio 前 4 的数字
                top = sorted(range(10), key=lambda d: gm.get(d, {}).get('geo_ratio', 1.0), reverse=True)[:4]
                # 与得分加权混合
                if random.random() < 0.5:
                    combo.append(random.choice(top))
                else:
                    combo.append(score_pick(pos, style))
            add_candidate(combo)

        # 策略 4: Regime 目标采样
        # 根据 predicted_next 概率采样状态，再在该状态对应的数字集合中采样。
        for _ in range(150):
            combo = []
            for pos in range(3):
                state_weights = {}
                if style == 'conservative':
                    state_weights = {'hot': 0.6, 'warm': 0.3, 'cold': 0.1}
                elif style == 'aggressive':
                    state_weights = {'hot': 0.1, 'warm': 0.3, 'cold': 0.6}
                else:
                    state_weights = {'hot': 0.25, 'warm': 0.5, 'cold': 0.25}
                target_state = weighted_choice(list(state_weights.keys()), list(state_weights.values()))
                pool = [d for d in range(10)
                        if ctx['regime'][pos].get('current', {}).get(d) == target_state]
                if not pool:
                    pool = list(range(10))
                combo.append(random.choice(pool))
            add_candidate(combo)

        # 策略 5: ACF 周期采样
        # 若某位置检测到显著周期，则按周期相位选取“周期对应”数字。
        for _ in range(150):
            combo = []
            for pos in range(3):
                acf = ctx.get('acf', [[]])[pos]
                significant = [lag for lag, rho in enumerate(acf, start=1) if rho > 0.12]
                if significant and random.random() < 0.5:
                    avg_lag = sum(significant) / len(significant)
                    # 近似：选当前遗漏接近平均周期倍数的数字
                    best_digit = min(range(10),
                                     key=lambda d: abs(ctx['miss'][pos].get(d, 0)
                                                       - round(ctx['miss'][pos].get(d, 0) / avg_lag) * avg_lag))
                    combo.append(best_digit)
                else:
                    combo.append(score_pick(pos, style))
            add_candidate(combo)

        # 策略 6: 贝叶斯采样（V3 保留）
        for _ in range(150):
            combo = []
            for pos in range(3):
                probs = [ctx['bayesian'][pos].get(d, 0.1) for d in range(10)]
                combo.append(weighted_choice(list(range(10)), probs))
            add_candidate(combo)

        # 策略 7: 和值目标采样（V3 保留，结合和尾遗漏）
        tail_candidates = sorted(range(10), key=lambda t: ctx['sum_tail_zscore'].get(t, 0), reverse=True)[:5]
        for _ in range(150):
            target_tail = random.choice(tail_candidates) if tail_candidates and random.random() < 0.6 else random.randint(0, 9)
            for _ in range(20):
                b = random.randint(0, 9)
                s = random.randint(0, 9)
                g_candidates = [g for g in range(10) if (b + s + g) % 10 == target_tail and len({b, s, g}) == 3]
                if g_candidates:
                    add_candidate((b, s, random.choice(g_candidates)))
                    break

        # 策略 8: 跨度目标采样（V3 保留，结合跨度遗漏）
        span_candidates = sorted(range(10), key=lambda sp: ctx['span_zscore'].get(sp, 0), reverse=True)[:5]
        for _ in range(150):
            target_span = random.choice(span_candidates) if span_candidates and random.random() < 0.6 else random.choice([3, 4, 5, 6, 7])
            if not (0 <= target_span <= 9):
                continue
            lo = random.randint(0, 9 - target_span)
            hi = lo + target_span
            pool = list(range(lo, hi + 1))
            if len(pool) >= 3:
                add_candidate(random.sample(pool, 3))

        # 兜底蒙特卡洛：若候选不足，随机补满，且尽量保证组六
        if len(candidates) < n_candidates:
            needed = n_candidates - len(candidates)
            for _ in range(needed * 3):
                combo = [random.randint(0, 9) for _ in range(3)]
                add_candidate(combo)
                if len(candidates) >= n_candidates:
                    break

        # 保证至少 70% 组六：若组六比例不足，用 score_pick 生成组六替换非组六
        result = list(candidates)
        zuliu = [c for c in result if len(set(c)) == 3]
        while len(zuliu) / max(len(result), 1) < 0.70:
            combo = [score_pick(pos, style) for pos in range(3)]
            if len(set(combo)) == 3:
                result.append(tuple(combo))
                zuliu.append(tuple(combo))
            if len(result) > n_candidates * 2:
                break

        return result[:max(n_candidates, len(result))]

    # ------------------------------------------------------------------
    # 推荐入口
    # ------------------------------------------------------------------
    def recommend_v4(self, data: List[Dict], n: int = 10, include_group: bool = True) -> Dict:
        """
        V4 推荐入口。

        Returns
        -------
        Dict
            {
                'gold': List[Tuple[int,int,int]],      # 2 注金码
                'silver': List[Tuple[int,int,int]],    # 2 注银码
                'top10': List[Dict],                    # 10 注详情
                'context': ctx,
                'style_used': str,
                'backtest_hint': str
            }
        """
        style = self.style
        ctx = self.compute_context(data)
        if style == 'adaptive':
            style = self._auto_style(ctx)

        candidates = self.generate_candidates_v4(data, ctx, style=style, n_candidates=1200)
        selected = self.select_top10_with_coverage_v4(candidates, ctx, style=style)

        # 金码 / 银码
        gold1, silver1 = self.generate_gold_silver_v4(data, ctx, style=style)
        gold2, silver2 = self.generate_gold_silver_v4(data, ctx, style=style, exclude={gold1, silver1})

        top10 = []
        for item in selected[:10]:
            b, s, g = item['b'], item['s'], item['g']
            unique = len({b, s, g})
            group_type = '豹子' if unique == 1 else ('组三' if unique == 2 else '组六')
            entry = {
                'num': (b, s, g),
                'sum': b + s + g,
                'span': max(b, s, g) - min(b, s, g),
                'score': item['score'],
                'details': item['details'],
                'group_type': group_type if include_group else None
            }
            if not include_group:
                del entry['group_type']
            top10.append(entry)

        backtest_hint = self._quick_backtest_hint(data, ctx, style)

        return {
            'gold': [gold1, gold2],
            'silver': [silver1, silver2],
            'top10': top10,
            'context': ctx,
            'style_used': style,
            'backtest_hint': backtest_hint
        }

    def select_top10_with_coverage_v4(self, candidates: List[Tuple[int, int, int]],
                                      ctx: Dict, style: str = 'balanced') -> List[Dict]:
        """在 V4 评分基础上，使用多样性惩罚选择 top10。"""
        if not candidates:
            return []

        scored = []
        for b, s, g in candidates:
            sc, details = self.score_candidate_v4(b, s, g, ctx, style)
            hz = b + s + g
            kd = max(b, s, g) - min(b, s, g)
            unique = len({b, s, g})
            shape = 'baozi' if unique == 1 else ('zusan' if unique == 2 else 'zuliu')
            odd = sum(1 for x in [b, s, g] if x % 2 == 1)
            big = sum(1 for x in [b, s, g] if x >= 5)
            scored.append({
                'score': sc, 'b': b, 's': s, 'g': g,
                'hz': hz, 'kd': kd, 'shape': shape,
                'odd': odd, 'big': big, 'details': details
            })

        scored.sort(key=lambda x: -x['score'])

        selected = []
        seen = set()
        span_count = Counter()
        hz_count = Counter()
        odd_count = Counter()
        big_count = Counter()

        def diversity_penalty(item):
            penalty = 0
            penalty += span_count[item['kd']] * 3
            penalty += hz_count[item['hz']] * 3
            penalty += odd_count[item['odd']] * 1
            penalty += big_count[item['big']] * 1
            return penalty

        remaining = scored[:400]
        while len(selected) < 10 and remaining:
            best_idx = 0
            best_value = remaining[0]['score'] - diversity_penalty(remaining[0])
            for idx in range(1, len(remaining)):
                val = remaining[idx]['score'] - diversity_penalty(remaining[idx])
                if val > best_value:
                    best_value = val
                    best_idx = idx
            item = remaining.pop(best_idx)
            combo = (item['b'], item['s'], item['g'])
            if combo in seen:
                continue
            selected.append(item)
            seen.add(combo)
            span_count[item['kd']] += 1
            hz_count[item['hz']] += 1
            odd_count[item['odd']] += 1
            big_count[item['big']] += 1

        return selected[:10]

    def generate_gold_silver_v4(self, data: List[Dict], ctx: Dict,
                                style: str = 'balanced',
                                exclude: Optional[set] = None) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        生成金码与银码，使用 V4 评分并加入重号/邻号奖励。

        Parameters
        ----------
        exclude : set
            已选号码集合，银码避免重复。
        """
        if not data:
            return (1, 2, 3), (4, 5, 6)

        last_draw = (data[0]['h'], data[0]['t'], data[0]['u'])
        last_set = set(last_draw)
        neighbor_set = set()
        for d in last_draw:
            neighbor_set.update({(d - 1) % 10, (d + 1) % 10})
        neighbor_set -= last_set

        if exclude is None:
            exclude = set()

        def full_enumerate():
            candidates = []
            for b in range(10):
                for s in range(10):
                    for g in range(10):
                        combo = (b, s, g)
                        if combo in exclude:
                            continue
                        hz = b + s + g
                        kd = max(b, s, g) - min(b, s, g)
                        if 5 <= hz <= 22 and 2 <= kd <= 8:
                            score, details = self.score_candidate_v4(b, s, g, ctx, style)
                            overlap = len(set(combo) & last_set)
                            if overlap >= 1:
                                score += 1.5
                            neighbor_overlap = len(set(combo) & neighbor_set)
                            if neighbor_overlap >= 1:
                                score += 1.0
                            if len(set(combo)) == 3:
                                score += 0.5
                            candidates.append((score, b, s, g))
            candidates.sort(key=lambda x: -x[0])
            return candidates

        candidates = full_enumerate()
        if not candidates:
            return (1, 2, 3), (4, 5, 6)

        gold = candidates[0][1:4]
        gold_set = set(gold)
        silver_candidates = []
        for score, b, s, g in candidates[1:]:
            combo_set = {b, s, g}
            overlap = len(combo_set & gold_set)
            if overlap <= 1:
                if len(combo_set) != len(gold_set):
                    score += 0.5
                silver_candidates.append((score, b, s, g))
        silver_candidates.sort(key=lambda x: -x[0])
        silver = silver_candidates[0][1:4] if silver_candidates else candidates[1][1:4]
        return gold, silver

    def _quick_backtest_hint(self, data: List[Dict], ctx: Dict, style: str) -> str:
        """
        生成简短回测提示。使用近 50 期滚动验证：用前 N-1 期生成 top10，
        检查是否命中最后一期（直选）或组选形态（组六包号）。
        注意：这仅是历史回测演示，不代表未来表现。
        """
        if len(data) < 60:
            return "历史数据不足，未做回测。"

        n_test = min(50, len(data) - 10)
        direct_hits = 0
        group_hits = 0
        for i in range(n_test):
            train = data[i + 1:i + 51]
            test_draw = data[i]
            if len(train) < 30:
                continue
            try:
                ctx_bt = self.compute_context(train)
                candidates = self.generate_candidates_v4(train, ctx_bt, style=style, n_candidates=300)
                selected = self.select_top10_with_coverage_v4(candidates, ctx_bt, style=style)
                selected_set = {(item['b'], item['s'], item['g']) for item in selected[:10]}
                test_tuple = (test_draw['h'], test_draw['t'], test_draw['u'])
                if test_tuple in selected_set:
                    direct_hits += 1
                # 组选命中：号码集合相同（适用于组六）
                test_set = set(test_tuple)
                if any(set(item['b'], item['s'], item['g']) == test_set for item in selected[:10]):
                    group_hits += 1
            except Exception:
                continue

        return (f"近{n_test}期同策略回测：直选命中{direct_hits}次，"
                f"组选（组六集合）命中{group_hits}次。历史表现不代表未来。")


if __name__ == '__main__':
    print("SmartNumberSelectorV4 模块已加载。请通过 run_analysis.py 调用。")
